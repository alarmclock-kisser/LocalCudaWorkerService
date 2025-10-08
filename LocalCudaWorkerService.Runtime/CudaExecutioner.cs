using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Runtime
{
	public class CudaExecutioner : IDisposable
	{
		private PrimaryContext CTX;
		private CudaRegister Register;
		private CudaFourier Fourier;
		private CudaCompiler Compiler;
		private CudaKernel? Kernel => this.Compiler.Kernel;

		public CudaExecutioner(PrimaryContext ctx, CudaRegister Register, CudaFourier fourier, CudaCompiler compiler)
		{
			this.CTX = ctx;
			this.Register = Register;
			this.Fourier = fourier;
			this.Compiler = compiler;
		}

		public void Dispose() { }

		// Safe device-to-device copy wrapper using Driver API.
		private static void CopyDeviceToDevice(CUdeviceptr dst, CUdeviceptr src, long byteSize)
		{
			var res = ManagedCuda.DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoD_v2(dst, src, (SizeT) byteSize);
			if (res != CUResult.Success)
			{
				throw new CudaException(res);
			}
		}

		// Heuristic detection whether kernel is the "merged 2D" variant (expects chunkCount as last argument)
		private static bool IsMerged2DVariant(string kernelName)
		{
			// Known merged kernels or pattern containing "double03" etc.
			if (string.IsNullOrWhiteSpace(kernelName))
			{
				return false;
			}

			kernelName = kernelName.ToLowerInvariant();
			return kernelName.Contains("double03") || kernelName.Contains("_merged") || kernelName.Contains("_2d") || kernelName.EndsWith("03");
		}

		// Kernels that operate per single FFT frame (need loop over chunks). Typical names contain complexes / pvoc / stateptr etc.
		private static bool IsPerFrameVariant(string kernelName)
		{
			if (string.IsNullOrWhiteSpace(kernelName))
			{
				return false;
			}

			kernelName = kernelName.ToLowerInvariant();
			return kernelName.Contains("complex") || kernelName.Contains("pvoc") || kernelName.Contains("stateptr");
		}

		// Try to set global phase state pointers if present (best effort, non-fatal on failure)
		private void TryInitializeGlobalPhaseState(int channels, int chunkSize)
		{
			try
			{
				if (this.Kernel == null)
				{
					return;
				}

				var name = this.Compiler.KernelName?.ToLowerInvariant() ?? string.Empty;
				if (!name.Contains("pvoc") || name.Contains("stateptr"))
				{
					return;
				}

				long bins = (long) channels * chunkSize;
				if (bins <= 0)
				{
					return;
				}

				IntPtr len = (nint) bins;
				var prev = this.Register.AllocateSingle<float>(len);
				var accum = this.Register.AllocateSingle<float>(len);
				if (prev == null || accum == null)
				{
					return;
				}

				// Füllen mit Null: einfache host-seitige Zero-Puffer-Kopie
				byte[] zero = new byte[bins * sizeof(float)];
				unsafe
				{
					fixed (byte* zp = zero)
					{
						var res1 = ManagedCuda.DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(prev.DevicePointers[0], (IntPtr) zp, (SizeT) zero.Length);
						var res2 = ManagedCuda.DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyHtoD_v2(accum.DevicePointers[0], (IntPtr) zp, (SizeT) zero.Length);
						if (res1 != CUResult.Success || res2 != CUResult.Success)
						{
							CudaService.Log("Phase state zero fill failed (non-fatal)");
						}
					}
				}
			}
			catch (Exception ex)
			{
				CudaService.Log(ex, "Phase state init attempt failed (non-fatal)");
			}
		}

		private IntPtr ExecutePerFrame(IntPtr fftIndexPtr, CudaMem fftMem, string kernelName, double factor, int chunkSize, int overlapSize, int sampleRate, int channels)
		{
			var k = this.Kernel!;
			int totalBins = chunkSize * channels; // if channels==1 falls back to chunkSize
			uint blockSize = 256;
			uint gridSize = (uint) ((totalBins + blockSize - 1) / blockSize);
			k.BlockDimensions = new dim3(blockSize, 1, 1);
			k.GridDimensions = new dim3(gridSize, 1, 1);

			// Allocate one reusable output frame buffer
			var tempOut = this.Register.AllocateSingle<float2>((nint) chunkSize);
			if (tempOut == null)
			{
				CudaService.Log("Failed to allocate temporary output frame buffer.");
				return fftIndexPtr;
			}

			// Initialize global phase state if needed (pvoc global variant)
			this.TryInitializeGlobalPhaseState(channels, chunkSize);

			for (int i = 0; i < fftMem.Count; i++)
			{
				var inPtr = fftMem.DevicePointers[i];
				var outPtr = tempOut.DevicePointers[0];
				object[] argsFrame;
				if (kernelName.ToLowerInvariant().Contains("stateptr"))
				{
					// Future hook: if we later add explicit stateptr arguments (prevPhase, phaseAccum) they'd be appended here.
					argsFrame = [inPtr, outPtr, chunkSize, overlapSize, sampleRate, channels, factor];
				}
				else
				{
					// Standard per-frame signature (input, output, chunkSize, overlapSize, samplerate, channels, factor)
					argsFrame = [inPtr, outPtr, chunkSize, overlapSize, sampleRate, channels, factor];
				}
				try
				{
					k.Run(argsFrame);
				}
				catch (Exception ex)
				{
					CudaService.Log(ex, $"Per-frame kernel launch failed at chunk {i}.");
					break;
				}
				// Copy output back to original chunk buffer
				CopyDeviceToDevice(inPtr, outPtr, chunkSize * sizeof(float) * 2);
			}

			this.Register.FreeMemory(tempOut);
			return fftIndexPtr;
		}

		private IntPtr ExecuteMerged2D(IntPtr fftIndexPtr, CudaMem fftMem, double factor, int chunkSize, int overlapSize, int sampleRate)
		{
			var contiguousIn = this.Register.AllocateSingle<float2>((nint) ((long) fftMem.Count * chunkSize));
			var contiguousOut = this.Register.AllocateSingle<float2>((nint) ((long) fftMem.Count * chunkSize));
			if (contiguousIn == null || contiguousOut == null)
			{
				CudaService.Log("Failed to allocate contiguous buffers.");
				return fftIndexPtr;
			}
			try
			{
				// Pack
				for (int i = 0; i < fftMem.Count; i++)
				{
					long byteSize = chunkSize * sizeof(float) * 2;
					long byteOffset = i * byteSize;
					CUdeviceptr dstBase = new(contiguousIn.IndexPointer);
					CUdeviceptr dstPtr = new(dstBase.Pointer + byteOffset);
					CopyDeviceToDevice(dstPtr, fftMem.DevicePointers[i], byteSize);
				}
				var k = this.Kernel!;
				int bins = chunkSize;
				uint bx = 32; uint by = 4;
				uint gx = (uint) ((bins + bx - 1) / bx);
				uint gy = (uint) ((fftMem.Count + by - 1) / by);
				k.BlockDimensions = new dim3(bx, by);
				k.GridDimensions = new dim3(gx, gy);
				object[] args =
				[
					new CUdeviceptr(contiguousIn.IndexPointer),
					new CUdeviceptr(contiguousOut.IndexPointer),
					chunkSize,
					overlapSize,
					sampleRate,
					factor,
					fftMem.Count
				];
				k.Run(args);
				// Unpack
				for (int i = 0; i < fftMem.Count; i++)
				{
					long byteSize = chunkSize * sizeof(float) * 2;
					long byteOffset = i * byteSize;
					CUdeviceptr srcBase = new(contiguousOut.IndexPointer);
					CUdeviceptr srcPtr = new(srcBase.Pointer + byteOffset);
					CopyDeviceToDevice(fftMem.DevicePointers[i], srcPtr, byteSize);
				}
			}
			catch (Exception ex)
			{
				CudaService.Log(ex, "Error executing merged time stretch kernel.");
			}
			finally
			{
				this.Register.FreeMemory(contiguousIn);
				this.Register.FreeMemory(contiguousOut);
			}
			return fftIndexPtr;
		}

		// Flexible ExecuteTimeStretch supporting both merged (2D) and per-frame kernels.
		public IntPtr ExecuteTimeStretch(IntPtr indexPointer, string kernel, double factor, int chunkSize, int overlapSize, int sampleRate, int channels, bool keep = false)
		{
			this.Compiler.LoadKernel(kernel);
			if (this.Kernel == null)
			{
				CudaService.Log("Kernel not loaded or invalid.");
				return indexPointer;
			}

			var mem = this.Register[indexPointer];
			if (mem == null || mem.IndexPointer == IntPtr.Zero || mem.IndexLength == IntPtr.Zero)
			{
				CudaService.Log("Memory not found or invalid pointer.");
				return IntPtr.Zero;
			}
			CudaService.Log($"Localized input memory: {mem.Count} chunks, total length: {mem.TotalLength}, total size: {mem.TotalSize} bytes.");

			bool transformed = false;
			IntPtr fftIndexPtr = mem.IndexPointer;
			if (mem.ElementType != typeof(float2))
			{
				fftIndexPtr = this.Fourier.PerformFft(indexPointer, keep);
				if (fftIndexPtr == IntPtr.Zero)
				{
					CudaService.Log("Failed to perform FFT on memory.");
					return indexPointer;
				}
				transformed = true;
				CudaService.Log("Memory transformed to float2 format for time stretch.");
			}

			var fftMem = this.Register[fftIndexPtr];
			if (fftMem == null || fftMem.IndexPointer == IntPtr.Zero || fftMem.IndexLength == IntPtr.Zero || fftMem.ElementType != typeof(float2))
			{
				CudaService.Log("FFT memory invalid.");
				return IntPtr.Zero;
			}

			int chunkCount = fftMem.Count;
			if (chunkCount <= 0)
			{
				CudaService.Log("No chunks to process.");
				return indexPointer;
			}
			if (chunkSize <= 0)
			{
				chunkSize = (int) fftMem.Lengths[0];
			}

			bool uniform = fftMem.Lengths.All(l => l.ToInt64() == fftMem.Lengths[0].ToInt64());
			if (!uniform || fftMem.Lengths[0].ToInt64() != chunkSize)
			{
				CudaService.Log("Non-uniform chunk sizes detected; Variant A expects uniform chunkSize.");
				return indexPointer;
			}

			var kernelName = kernel;
			bool merged = IsMerged2DVariant(kernelName);
			bool perFrame = IsPerFrameVariant(kernelName);
			if (!merged && !perFrame)
			{
				// Fallback: try merged if only one of the signatures matches length of argument guess
				// Default to merged for backward compatibility
				merged = true;
			}

			CudaService.Log($"Kernel style detected: {(merged ? "merged-2D" : "per-frame")} for '{kernelName}'.");

			try
			{
				if (merged)
				{
					fftIndexPtr = this.ExecuteMerged2D(fftIndexPtr, fftMem, factor, chunkSize, overlapSize, sampleRate);
				}
				else
				{
					fftIndexPtr = this.ExecutePerFrame(fftIndexPtr, fftMem, kernelName, factor, chunkSize, overlapSize, sampleRate, channels);
				}
			}
			catch (Exception ex)
			{
				CudaService.Log(ex, "Time stretch execution failed.");
			}

			IntPtr outIndexPtr = fftIndexPtr;
			if (transformed)
			{
				outIndexPtr = this.Fourier.PerformIfft(fftIndexPtr, keep);
				if (outIndexPtr == IntPtr.Zero)
				{
					CudaService.Log("Failed to perform inverse FFT on output.");
					return indexPointer;
				}
				CudaService.Log("Transformed output memory back to float format after time stretch.");
			}

			var outMem = this.Register[outIndexPtr];
			if (outMem == null || outMem.IndexPointer == IntPtr.Zero)
			{
				CudaService.Log("Output memory invalid after execution.");
				return indexPointer;
			}
			CudaService.Log($"Output memory after execution: {outMem.Count} chunks, total length: {outMem.TotalLength}, total size: {outMem.TotalSize} bytes.");
			return outIndexPtr;
		}

		public IntPtr ExecuteGenericAudioKernel(IntPtr indexPointer, string kernel, int chunkSize, int oberlapSize, int sampleRate, int channels = 0, Dictionary<string, object>? additionalArgs = null)
		{
			this.Compiler.LoadKernel(kernel);
			if (this.Kernel == null)
			{
				CudaService.Log("Kernel not loaded or invalid.");
				return indexPointer;
			}
			var mem = this.Register[indexPointer];
			if (mem == null || mem.IndexPointer == IntPtr.Zero || mem.IndexLength == IntPtr.Zero)
			{
				CudaService.Log("Memory not found or invalid pointer.");
				return IntPtr.Zero;
			}
			CudaService.Log($"Localized input memory: {mem.Count} chunks, total length: {mem.TotalLength}, total size: {mem.TotalSize} bytes.");

			var paramList = this.SortKernelParameters(kernel, indexPointer, chunkSize, oberlapSize, sampleRate, channels, additionalArgs ?? []);
			if (paramList.Length == 0)
			{
				CudaService.Log("No parameters sorted for kernel execution.");
				return indexPointer;
			}

			try
			{
				this.Kernel.Run(paramList);
			}
			catch (Exception ex)
			{
				CudaService.Log(ex, "Generic kernel execution failed.");
				return indexPointer;
			}


			return indexPointer;
		}

		public object[] SortKernelParameters(string kernel, IntPtr dataPointer, int chunkSize, int overlapSize, int sampleRate, int channels, Dictionary<string, object> additionalArgs)
		{
			string cuPath = this.Compiler.GetCuFiles().FirstOrDefault(p => Path.GetFileNameWithoutExtension(p).Equals(kernel, StringComparison.OrdinalIgnoreCase)) ?? string.Empty;
			if (string.IsNullOrWhiteSpace(cuPath))
			{
				CudaService.Log("Kernel .cu file not found	 for parameter sorting.");
				return [];
			}

			this.Compiler.LoadKernel(cuPath);
			var paramList = new List<object>();
			var k = this.Kernel!;
			var parameters = this.Compiler.GetArguments();
			foreach (var p in parameters)
			{
				string pName = p.Key?.ToLowerInvariant() ?? string.Empty;
				Type pType = p.Value;
				if (pType == typeof(CUdeviceptr))
				{
					if (pName.Contains("input"))
					{
						var mem = this.Register[dataPointer];
						if (mem == null || mem.IndexPointer == IntPtr.Zero)
						{
							CudaService.Log("Memory not found or invalid pointer for input.");
							paramList.Add(new CUdeviceptr(0));
						}
						else
						{
							paramList.Add(mem.DevicePointers[0]);
						}
					}
					else if (pName.Contains("output"))
					{
						var mem = this.Register[dataPointer];
						if (mem == null || mem.IndexPointer == IntPtr.Zero)
						{
							CudaService.Log("Memory not found or invalid pointer for output.");
							paramList.Add(new CUdeviceptr(0));
						}
						else
						{
							paramList.Add(mem.DevicePointers[0]);
						}
					}
					else
					{
						paramList.Add(new CUdeviceptr(0));
					}
				}
				else if (pType == typeof(int))
				{
					if (pName.Contains("chunksize"))
					{
						paramList.Add(chunkSize);
					}
					else if (pName.Contains("overlapsize"))
					{
						paramList.Add(overlapSize);
					}
					else if (pName.Contains("samplerate"))
					{
						paramList.Add(sampleRate);
					}
					else if (pName.Contains("channels"))
					{
						paramList.Add(channels);
					}
					else
					{
						paramList.Add(0);
					}
				}
				else if (pType == typeof(float) || pType == typeof(double))
				{
					if (additionalArgs != null && additionalArgs.TryGetValue(p.Key ?? string.Empty, out var value))
					{
						if (value is float fVal && pType == typeof(float))
						{
							paramList.Add(fVal);
						}
						else if (value is double dVal && pType == typeof(double))
						{
							paramList.Add(dVal);
						}
						else if (value is int iVal)
						{
							paramList.Add(Convert.ChangeType(iVal, pType));
						}
						else
						{
							paramList.Add(0);
						}
					}
					else
					{
						paramList.Add(0);
					}
				}
				else
				{
					paramList.Add(null!);
				}
			}

			CudaService.Log($"Sorted kernel parameters for '{kernel}': {paramList.Count} parameters.");
			CudaService.Log($"Parameter list: {string.Join(", ", paramList.Select(p => p?.ToString() ?? "null"))}");

			return paramList.ToArray();
		}

		// Legacy async API bridges (Variant A uses single launch). Keep signatures expected by CudaService.
		public Task<IntPtr> ExecuteTimeStretchAsync(IntPtr pointer, string kernel, double factor, int chunkSize, int overlapSize, int sampleRate, int channels, bool asMany = false, bool keep = false)
		{
			return Task.Run(() => this.ExecuteTimeStretch(pointer, kernel, factor, chunkSize, overlapSize, sampleRate, channels, keep));
		}

		public Task<IntPtr> ExecuteTimeStretchInterleavedAsync(IntPtr pointer, string kernel, double factor, int chunkSize, int overlapSize, int sampleRate, int channels, int maxStreams = 1, bool asMany = false, bool keep = false)
		{
			return Task.Run(() => this.ExecuteTimeStretch(pointer, kernel, factor, chunkSize, overlapSize, sampleRate, channels, keep));
		}

		public Task<IntPtr> ExecuteGenericAudioKernelAsync(IntPtr pointer, string kernel, int chunkSize, int overlapSize, int sampleRate, int channels = 0, Dictionary<string, object>? additionalArgs = null)
		{
			return Task.Run(() => this.ExecuteGenericAudioKernel(pointer, kernel, chunkSize, overlapSize, sampleRate, channels, additionalArgs));
		}



		// Generic exec
		public TResult[] ExecuteGenericKernelSingle<TResult>(string kernelCode, object[]? inputData, string? inputDataType = "Byte", long outputElementCount = 0, Dictionary<string, string>? arguments = null, int workDimensions = 1, bool freeInput = true, bool freeOutput = true) where TResult : unmanaged
		{
			if (string.IsNullOrEmpty(kernelCode) || this.Register == null || outputElementCount <= 0 || workDimensions < 1 || workDimensions > 3)
			{
				return [];
			}

			TResult[] result = [];

			// Try to compile the kernel
			CudaKernel? kernel = this.Compiler.CompileLoadKernelFromString(kernelCode);
			if (kernel == null)
			{
				Console.WriteLine("CUDA-EXEC| Kernel compilation or loading failed.");
				return [];
			}

			CudaMem? inputMem = null;
			if (inputData is { LongLength: > 0 } && !string.IsNullOrEmpty(inputDataType))
			{
				inputDataType = inputDataType.ToLowerInvariant().Trim();

				// WICHTIG: Cast<object>() kommt von boxed ValueTypes (ConvertStringToTypeAsync)
				inputMem = inputDataType switch
				{
					"byte" or "uint8" or "uchar" => this.Register.PushData<byte>(inputData.Cast<byte>().ToArray()),
					"sbyte" or "int8" or "char" => this.Register.PushData<sbyte>(inputData.Cast<sbyte>().ToArray()),
					"short" or "int16" => this.Register.PushData<short>(inputData.Cast<short>().ToArray()),
					"ushort" or "uint16" => this.Register.PushData<ushort>(inputData.Cast<ushort>().ToArray()),
					"int" or "int32" => this.Register.PushData<int>(inputData.Cast<int>().ToArray()),
					"uint" or "uint32" => this.Register.PushData<uint>(inputData.Cast<uint>().ToArray()),
					"long" or "int64" => this.Register.PushData<long>(inputData.Cast<long>().ToArray()),
					"ulong" or "uint64" => this.Register.PushData<ulong>(inputData.Cast<ulong>().ToArray()),
					"float" or "single" => this.Register.PushData<float>(inputData.Cast<float>().ToArray()),
					"double" => this.Register.PushData<double>(inputData.Cast<double>().ToArray()),
					"float2" or "complex" => this.Register.PushData<float2>(inputData.Cast<float2>().ToArray()),
					"intptr" or "pointer" => this.Register.PushData<IntPtr>(inputData.Cast<IntPtr>().ToArray()),
					_ => null,
				};

				Console.WriteLine($"CUDA-EXEC| Input Memory: {inputMem?.TotalLength} elements of type {inputMem?.ElementType.Name}");
			}

			// FIX: outputElementCount AllocateSingle<T> macht schon intern elementSize * count
			CudaMem? outputMem = this.Register.AllocateSingle<TResult>((nint) outputElementCount);
			if (outputMem == null || outputMem.Count <= 0)
			{
				Console.WriteLine("CUDA-EXEC| Failed to allocate output memory.");
				return [];
			}

			Console.WriteLine($"CUDA-EXEC| Output Memory: {outputMem.TotalLength} elements of type {outputMem.ElementType.Name}");

			// Argumente zusammenführen (Pointer richtig setzen)
			object[] mergedArgs = this.MergeGenericKernelArgumentsDynamic(kernelCode, inputMem?.DevicePointers.FirstOrDefault(), outputMem.DevicePointers.FirstOrDefault(), arguments);
			// Sicherheits-Check: Kernel-Argumentanzahl
			if (mergedArgs.Length == 0)
			{
				Console.WriteLine("CUDA-EXEC| No kernel arguments found.");
				return [];
			}

			// Dimensions
			long elementsTotal = outputElementCount;

			// Work dimensions
			uint workDim = (uint) workDimensions;

			UIntPtr[] globalWorkSize = workDim switch
			{
				1 => [(UIntPtr) elementsTotal],
				2 => [(UIntPtr) Math.Ceiling(Math.Sqrt(elementsTotal)), (UIntPtr) Math.Ceiling(Math.Sqrt(elementsTotal))],
				3 =>
				[
					(UIntPtr) Math.Ceiling(Math.Pow(elementsTotal, 1.0 / 3.0)),
					(UIntPtr) Math.Ceiling(Math.Pow(elementsTotal, 1.0 / 3.0)),
					(UIntPtr) Math.Ceiling(Math.Pow(elementsTotal, 1.0 / 3.0))
				],
				_ => [(UIntPtr) elementsTotal]
			};

			Console.WriteLine("CUDA-EXEC| Elements Total: " + elementsTotal + ", WorkDim: " + workDim + ", GlobalWorkSize: " + string.Join(", ", globalWorkSize.Select(g => g.ToString())));

			// Set block and grid sizes (1|2|3D depending on workDim and work sizes)
			kernel.BlockDimensions = workDim switch
			{
				1 => new dim3(256, 1, 1),
				2 => new dim3(16, 16, 1),
				3 => new dim3(8, 8, 8),
				_ => new dim3(256, 1, 1)
			};
			kernel.GridDimensions = workDim switch
			{
				1 => new dim3((uint) ((elementsTotal + kernel.BlockDimensions.x - 1) / kernel.BlockDimensions.x), 1, 1),
				2 => new dim3((uint) ((globalWorkSize[0].ToUInt64() + kernel.BlockDimensions.x - 1) / kernel.BlockDimensions.x),
							  (uint) ((globalWorkSize[1].ToUInt64() + kernel.BlockDimensions.y - 1) / kernel.BlockDimensions.y), 1),
				3 => new dim3((uint) ((globalWorkSize[0].ToUInt64() + kernel.BlockDimensions.x - 1) / kernel.BlockDimensions.x),
							  (uint) ((globalWorkSize[1].ToUInt64() + kernel.BlockDimensions.y - 1) / kernel.BlockDimensions.y),
							  (uint) ((globalWorkSize[2].ToUInt64() + kernel.BlockDimensions.z - 1) / kernel.BlockDimensions.z)),
				_ => new dim3((uint) ((elementsTotal + kernel.BlockDimensions.x - 1) / kernel.BlockDimensions.x), 1, 1)
			};

			// EXEC
			kernel.Run(mergedArgs);

			// Ergebnis zurückholen
			if (outputMem.ElementType != typeof(TResult))
			{
				Console.WriteLine("CUDA-EXEC| Warning: Output memory type mismatch. Expected " + typeof(TResult).Name + " but got " + outputMem.ElementType.Name);
			}

			result = this.Register.PullData<TResult>(outputMem.IndexPointer);
			if (result == null || result.LongLength == 0)
			{
				Console.WriteLine("CUDA-EXEC| Failed to retrieve output data from device.");
				result = [];
			}

			// Aufräumen
			long freed = 0;
			if (inputMem != null && freeInput)
			{
				freed = this.Register.FreeMemory(inputMem);
				Console.WriteLine("CUDA-EXEC| Freed input memory: " + freed + " bytes.");
			}
			if (outputMem != null && freeOutput)
			{
				freed = this.Register.FreeMemory(outputMem);
				Console.WriteLine("CUDA-EXEC| Freed output memory: " + freed + " bytes.");
			}

			Console.WriteLine("CUDA-EXEC| Kernel execution completed. Retrieved " + result.LongLength + " elements.");
			this.CTX.Synchronize();

			return result;
		}

		public async Task<TResult[]> ExecuteGenericKernelSingleAsync<TResult>(string kernelCode, object[]? inputData, string? inputDataType = "Byte", long outputElementCount = 0, Dictionary<string, string>? arguments = null, int workDimensions = 1) where TResult : unmanaged
		{
			return await Task.Run(() => this.ExecuteGenericKernelSingle<TResult>(kernelCode, inputData, inputDataType, outputElementCount, arguments, workDimensions));
		}





		public object[] MergeGenericKernelArgumentsDynamic(string kernelCode, CUdeviceptr? inputBuffer = null, CUdeviceptr? outputBuffer = null, Dictionary<string, string>? arguments = null)
		{
			if (kernelCode == null)
			{
				return [];
			}

			var requiredArgs = this.Compiler.GetArguments(kernelCode);
			if (requiredArgs == null || requiredArgs.Count == 0)
			{
				return [];
			}

			object[] sortedArgs = new object[requiredArgs.Count];

			for (int i = 0; i < requiredArgs.Count; i++)
			{
				string argName = requiredArgs.ElementAt(i).Key;
				Type argType = requiredArgs.ElementAt(i).Value;
				string argNameLower = argName.ToLowerInvariant();
				bool isPointer = argType.Name.EndsWith("*");

				// LOG
				Console.WriteLine("CUDA-EXEC| Merging Argument: " + argType.Name + " " + argName);
				if (isPointer)
				{
					if ((argNameLower.Contains("in") || argNameLower == "input") && inputBuffer != null)
					{
						sortedArgs[i] = inputBuffer.Value;
						Console.WriteLine("CUDA-EXEC| Using Input Buffer for Argument: " + argName);
						continue;
					}
					if ((argNameLower.Contains("out") || argNameLower == "output") && outputBuffer != null)
					{
						sortedArgs[i] = outputBuffer.Value;
						Console.WriteLine("CUDA-EXEC| Using Output Buffer for Argument: " + argName);
						continue;
					}
					sortedArgs[i] = new CUdeviceptr();
					Console.WriteLine("CUDA-EXEC| No Buffer Provided for Argument: " + argName + ", using empty CLBuffer");
					continue;
				}

				if (arguments != null && arguments.TryGetValue(argName, out string? raw))
				{
					sortedArgs[i] = raw == null ? 0 : ParseScalar(argType, raw);
					Console.WriteLine("CUDA-EXEC| Using Provided Value for Argument: " + argName + " = " + (sortedArgs[i]?.ToString() ?? "null"));
				}
				else
				{
					sortedArgs[i] = argType == typeof(int) ? 0 :
									argType == typeof(long) ? 0L :
									argType == typeof(float) ? 0f :
									argType == typeof(double) ? 0d :
									argType == typeof(byte) ? (byte) 0 :
									argType == typeof(uint) ? 0u :
									0;
					Console.WriteLine("CUDA-EXEC| No Value Provided for Argument: " + argName + ", using default = " + (sortedArgs[i]?.ToString() ?? "null") + " of type " + argType.Name);
				}
			}

			// Absicherung: keine null / unbekannten Typen an Kernel geben
			for (int i = 0; i < sortedArgs.Length; i++)
			{
				if (sortedArgs[i] == null)
				{
					Console.WriteLine("CUDA-EXEC| Warning: Argument " + i + " (" + requiredArgs.ElementAt(i).Key + ") is null, replacing with 0");
					sortedArgs[i] = 0;
				}
				else
				{
					bool supported = sortedArgs[i] is CUdeviceptr
						|| sortedArgs[i] is int
						|| sortedArgs[i] is long
						|| sortedArgs[i] is float
						|| sortedArgs[i] is double
						|| sortedArgs[i] is byte
						|| sortedArgs[i] is uint
						|| sortedArgs[i] is bool
						|| sortedArgs[i] is float2
						|| sortedArgs[i] is IntPtr;
					if (!supported)
					{
						Console.WriteLine("CUDA-EXEC| Warning: Argument " + i + " (" + requiredArgs.ElementAt(i).Key + ") has unsupported type " + sortedArgs[i].GetType().Name + ", replacing with 0");
						sortedArgs[i] = 0;
					}
				}
			}

			// Print sorted args with Type:Name(value)
			string log = "CUDA-EXEC| Merged Kernel Arguments: " + Environment.NewLine;
			for (int i = 0; i < sortedArgs.Length; i++)
			{
				var arg = sortedArgs[i];
				string typeName = arg?.GetType().Name ?? "null";
				string valueStr = arg?.ToString() ?? "null";
				string argName = requiredArgs.ElementAt(i).Key;
				if (arg is CUdeviceptr ptr)
				{
					valueStr = ptr.Pointer != IntPtr.Zero ? ptr.Pointer.ToString() : "null";
				}
				log += $"[{i}]{typeName}:{argName}='{valueStr}', " + Environment.NewLine;
			}

			return sortedArgs;
		}

		private static object ParseScalar(Type t, string raw)
		{
			if (t == typeof(int) && int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int i))
			{
				return i;
			}

			if (t == typeof(long) && long.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out long l))
			{
				return l;
			}

			if (t == typeof(float) && float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out float f))
			{
				return f;
			}

			if (t == typeof(double) && double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out double d))
			{
				return d;
			}

			if (t == typeof(byte) && byte.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out byte b))
			{
				return b;
			}

			if (t == typeof(uint) && uint.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out uint u))
			{
				return u;
			}

			if (t == typeof(bool))
			{
				if (raw == "0")
				{
					return false;
				}

				if (raw == "1")
				{
					return true;
				}

				if (bool.TryParse(raw, out bool bo))
				{
					return bo;
				}

				return false;
			}
			return 0;
		}
	}
}
