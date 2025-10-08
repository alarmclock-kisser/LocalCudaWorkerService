using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.NVRTC;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Runtime
{
	public class CudaCompiler
	{
		// ----- ----- ATTRIBUTES ----- ----- \\
		private PrimaryContext Context;

		public CudaKernel? Kernel = null;
		public string? KernelName = null;
		public string? KernelFile = null;
		public string? KernelCode = null;


		public List<string> SourceFiles => this.GetCuFiles();
		public List<string> CompiledFiles => this.GetPtxFiles();


		internal string KernelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "LocalCudaWorkerService.Runtime", "Kernels");

		// ----- ----- CONSTRUCTORS ----- ----- \\
		public CudaCompiler(PrimaryContext context)
		{
			this.Context = context;

			// If deployed kernelpath does not exist like that
			if (!Directory.Exists(this.KernelPath))
			{
				this.KernelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Kernels");
				if (!Directory.Exists(this.KernelPath))
				{
					this.KernelPath = string.Empty;
				}
			}

			// Compile all kernels
			this.CompileAll(false, true);
		}



		// Dispose & Unload
		public void Dispose()
		{
			// Dispose of kernels
			this.UnloadKernel();
		}

		public void UnloadKernel()
		{
			// Unload kernel
			if (this.Kernel != null)
			{
				try
				{
					this.Context.UnloadKernel(this.Kernel);
				}
				catch (Exception ex)
				{
					CudaService.Log("Failed to unload kernel", ex.Message, 1);
				}
				this.Kernel = null;
			}
		}


		// Get files
		public List<string> GetPtxFiles(string? path = null)
		{
			path ??= Path.Combine(this.KernelPath, "PTX");

			// Get all PTX files in kernel path
			string[] files = Directory.GetFiles(path, "*.ptx").Select(f => Path.GetFullPath(f)).ToArray();

			// Return files
			return files.ToList();
		}

		public List<string> GetCuFiles(string? path = null)
		{
			path ??= Path.Combine(this.KernelPath, "CU");

			// Get all CU files in kernel path
			string[] files = Directory.GetFiles(path, "*.cu").Select(f => Path.GetFullPath(f)).ToArray();

			// Return files
			return files.ToList();
		}

		public string? SelectLatestKernel()
		{
			string[] files = this.CompiledFiles.ToArray();

			// Get file info (last modified), sort by last modified date, select latest
			FileInfo[] fileInfos = files.Select(f => new FileInfo(f)).OrderByDescending(f => f.LastWriteTime).ToArray();

			string latestFile = fileInfos.FirstOrDefault()?.FullName ?? "";
			string latestName = Path.GetFileNameWithoutExtension(latestFile) ?? "";
			if (string.IsNullOrEmpty(latestFile) || string.IsNullOrEmpty(latestName))
			{
				CudaService.Log("No compiled kernels found", "", 1);
				return null;
			}

			return latestName;
		}


		// Compile
		public void CompileAll(bool silent = false, bool logErrors = false)
		{
			List<string> sourceFiles = this.SourceFiles;

			// Compile all source files
			foreach (string sourceFile in sourceFiles)
			{
				string? ptx = this.CompileKernel(sourceFile, silent);
				if (string.IsNullOrEmpty(ptx) && logErrors)
				{
					CudaService.Log("Compilation failed: ", Path.GetFileNameWithoutExtension(sourceFile), 1);
				}
			}
		}

		public string? CompileKernel(string filepath, bool silent = false)
		{
			if (this.Context == null)
			{
				if (!silent)
				{
					CudaService.Log("No CUDA available", "", 1);
				}
				return null;
			}

			// If file is not a .cu file, but raw kernel string, compile that
			if (Path.GetExtension(filepath) != ".cu")
			{
				return this.CompileString(filepath, silent);
			}

			string kernelName = Path.GetFileNameWithoutExtension(filepath);

			string logpath = Path.Combine(this.KernelPath, "Logs", kernelName + ".log");

			Stopwatch sw = Stopwatch.StartNew();
			if (!silent)
			{
				CudaService.Log("Compiling kernel '" + kernelName + "'");
			}

			// Load kernel file
			string kernelCode = File.ReadAllText(filepath);


			CudaRuntimeCompiler rtc = new(kernelCode, kernelName);

			try
			{
				// Compile kernel
				rtc.Compile([]);
				string log = rtc.GetLogAsString();

				if (log.Length > 0)
				{
					// Count double \n
					int count = log.Split(["\n\n"], StringSplitOptions.None).Length - 1;
					if (!silent)
					{
						CudaService.Log("Compiled with warnings", count.ToString(), 1);
					}
					File.WriteAllText(logpath, log);
				}

				sw.Stop();
				long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
				if (!silent)
				{
					CudaService.Log("Compiled within " + deltaMicros + " µs", "Repo\\" + Path.GetRelativePath(this.KernelPath, logpath), 1);
				}

				// Get ptx code
				byte[] ptxCode = rtc.GetPTX();

				// Export ptx
				string ptxPath = Path.Combine(this.KernelPath, "PTX", kernelName + ".ptx");
				File.WriteAllBytes(ptxPath, ptxCode);

				if (!silent)
				{
					CudaService.Log("PTX exported", ptxPath, 1);
				}

				return ptxPath;
			}
			catch (Exception ex)
			{
				File.WriteAllText(logpath, rtc.GetLogAsString());
				CudaService.Log(ex.Message, ex.InnerException?.Message ?? "", 1);

				return rtc.GetLogAsString();
			}

		}

		public string? CompileString(string kernelString, bool silent = false)
		{
			if (this.Context == null)
			{
				if (!silent)
				{
					CudaService.Log("No CUDA available", "", 1);
				}
				return null;
			}

			string kernelName = kernelString.Split("void ")[1].Split("(")[0];

			string logpath = Path.Combine(this.KernelPath, "Logs", kernelName + ".log");

			Stopwatch sw = Stopwatch.StartNew();
			if (!silent)
			{
				CudaService.Log("Compiling kernel '" + kernelName + "'");
			}

			// Load kernel file
			string kernelCode = kernelString;

			// Save also the kernel string as .c file
			string cPath = Path.Combine(this.KernelPath, "CU", kernelName + ".cu");
			File.WriteAllText(cPath, kernelCode);


			CudaRuntimeCompiler rtc = new(kernelCode, kernelName);

			try
			{
				// Compile kernel
				rtc.Compile([]);
				string log = rtc.GetLogAsString();

				if (log.Length > 0)
				{
					// Count double \n
					int count = log.Split(["\n\n"], StringSplitOptions.None).Length - 1;
					if (!silent)
					{
						CudaService.Log("Compiled with warnings", count.ToString(), 1);
					}
					File.WriteAllText(logpath, rtc.GetLogAsString());
				}


				sw.Stop();
				long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
				if (!silent)
				{
					CudaService.Log("Compiled within " + deltaMicros + " µs", "Repo\\" + Path.GetRelativePath(this.KernelPath, logpath), 1);
				}


				// Get ptx code
				byte[] ptxCode = rtc.GetPTX();

				// Export ptx
				string ptxPath = Path.Combine(this.KernelPath, "PTX", kernelName + ".ptx");
				File.WriteAllBytes(ptxPath, ptxCode);

				if (!silent)
				{
					CudaService.Log("PTX exported", ptxPath, 1);
				}

				return ptxPath;
			}
			catch (Exception ex)
			{
				File.WriteAllText(logpath, rtc.GetLogAsString());
				CudaService.Log(ex.Message, ex.InnerException?.Message ?? "", 1);

				return null;
			}
		}

		public string? PrecompileKernelString(string kernelString, bool silent = false)
		{
			// Check contains "extern c"
			if (!kernelString.Contains("extern \"C\""))
			{
				if (!silent)
				{
					CudaService.Log("Kernel string does not contain 'extern \"C\"'", "", 1);
				}
				return null;
			}

			// Check contains "__global__ "
			if (!kernelString.Contains("__global__"))
			{
				if (!silent)
				{
					CudaService.Log("Kernel string does not contain '__global__'", "", 1);
				}
				return null;
			}

			// Check contains "void "
			if (!kernelString.Contains("void "))
			{
				if (!silent)
				{
					CudaService.Log("Kernel string does not contain 'void '", "", 1);
				}
				return null;
			}

			// Check contains int
			if (!kernelString.Contains("int ") && !kernelString.Contains("long "))
			{
				if (!silent)
				{
					CudaService.Log("Kernel string does not contain 'int ' (for array length)", "", 1);
				}
				return null;
			}

			// Check if every bracket is closed (even amount) for {} and () and []
			int open = kernelString.Count(c => c == '{');
			int close = kernelString.Count(c => c == '}');
			if (open != close)
			{
				if (!silent)
				{
					CudaService.Log("Kernel string has unbalanced brackets", " { } ", 1);
				}
				return null;
			}
			open = kernelString.Count(c => c == '(');
			close = kernelString.Count(c => c == ')');
			if (open != close)
			{
				if (!silent)
				{
					CudaService.Log("Kernel string has unbalanced brackets", " ( ) ", 1);
				}
				return null;
			}
			open = kernelString.Count(c => c == '[');
			close = kernelString.Count(c => c == ']');
			if (open != close)
			{
				if (!silent)
				{
					CudaService.Log("Kernel string has unbalanced brackets", " [ ] ", 1);
				}
				return null;
			}

			// Check if kernel contains "blockIdx.x" and "blockDim.x" and "threadIdx.x"
			if (!kernelString.Contains("blockIdx.x") || !kernelString.Contains("blockDim.x") || !kernelString.Contains("threadIdx.x"))
			{
				if (!silent)
				{
					CudaService.Log("Kernel string should contain 'blockIdx.x', 'blockDim.x' and 'threadIdx.x'", "", 2);
				}
			}

			// Get name between "void " and "("
			int start = kernelString.IndexOf("void ") + "void ".Length;
			int end = kernelString.IndexOf("(", start);
			string name = kernelString.Substring(start, end - start);

			// Trim every line ends from empty spaces (split -> trim -> aggregate)
			kernelString = kernelString.Split("\n").Select(x => x.TrimEnd()).Aggregate((x, y) => x + "\n" + y);

			// Log name
			if (!silent)
			{
				CudaService.Log("Succesfully precompiled kernel string", "Name: " + name, 1);
			}

			return name;
		}


		// Load
		public CudaKernel? LoadKernel(string kernelName, bool silent = false)
		{
			if (this.Context == null)
			{
				CudaService.Log("No CUDA context available", "", 1);
				return null;
			}

			// Unload?
			if (this.Kernel != null)
			{
				this.UnloadKernel();
			}

			// Get kernel path
			string kernelPath = Path.Combine(this.KernelPath, "PTX", kernelName + ".ptx");

			// Get log path
			string logpath = Path.Combine(this.KernelPath, "Logs", kernelName + ".log");

			// Log
			Stopwatch sw = Stopwatch.StartNew();
			if (!silent)
			{
				CudaService.Log("Started loading kernel " + kernelName);
			}

			// Try to load kernel
			try
			{
				// Load ptx code
				byte[] ptxCode = File.ReadAllBytes(kernelPath);

				string cuPath = Path.Combine(this.KernelPath, "CU", kernelName + ".cu");

				// Load kernel
				this.Kernel = this.Context.LoadKernelPTX(ptxCode, kernelName);
				this.KernelName = kernelName;
				this.KernelFile = kernelPath;
				this.KernelCode = File.ReadAllText(cuPath);
			}
			catch (Exception ex)
			{
				if (!silent)
				{
					CudaService.Log("Failed to load kernel " + kernelName, ex.Message, 1);
					string logMsg = ex.Message + Environment.NewLine + Environment.NewLine + ex.InnerException?.Message ?? "";
					File.WriteAllText(logpath, logMsg);
				}
				this.Kernel = null;
			}

			// Log
			sw.Stop();
			long deltaMicros = sw.ElapsedTicks / (Stopwatch.Frequency / (1000L * 1000L));
			if (!silent)
			{
				CudaService.Log("Kernel loaded within " + deltaMicros.ToString("N0") + " µs", "", 2);
			}

			return this.Kernel;
		}

		internal CudaKernel? CompileLoadKernelFromString(string kernelCode)
		{
			// Precompile kernel string (check for validity and get name)
			string? kernelName = this.PrecompileKernelString(kernelCode, false);
			if (string.IsNullOrEmpty(kernelName))
			{
				return null;
			}

			// Compile kernel string
			string? ptxPath = this.CompileString(kernelCode, false);
			if (ptxPath == null)
			{
				return null;
			}

			// Load kernel
			return this.LoadKernel(kernelName ?? "", false);
		}


		// Argument extraction
		// Hilfsfunktion zum Entfernen von Modifizierern
		private static string RemoveTypeModifiers(string typeName)
		{
			// Liste der zu entfernenden Modifizierer
			string[] modifiers = ["const", "__restrict__", "restrict", "__restrict", "__const__", "__volatile__", "volatile"];
			var parts = typeName.Split(' ', StringSplitOptions.RemoveEmptyEntries)
								.Where(p => !modifiers.Contains(p))
								.ToArray();
			return string.Join(" ", parts);
		}

		public Type GetArgumentType(string typeName)
		{
			string typeIdentifier = RemoveTypeModifiers(typeName).Split(' ').LastOrDefault()?.Trim() ?? "object";
			bool isPointer = typeIdentifier.EndsWith("*");

			Type type = typeIdentifier switch
			{
				"int" => typeof(int),
				"float" => typeof(float),
				"double" => typeof(double),
				"char" => typeof(char),
				"bool" => typeof(bool),
				"void" => typeof(void),
				"byte" => typeof(byte),
				"float2" => typeof(System.Numerics.Vector2),
				_ => typeof(void)
			};

			CudaService.Log($"Argument type: {typeIdentifier} -> {this.GetArgumentTypeString(typeName)}", "", 1);

			if (isPointer)
			{
				// Get pointer type from identifier DO NOT JUST RETURN INTPTR
				type = typeIdentifier.ToLower() switch
				{
					"int*" => typeof(int*),
					"float*" => typeof(float*),
					"double*" => typeof(double*),
					"char*" => typeof(char*),
					"bool*" => typeof(bool*),
					"byte*" => typeof(byte*),
					"float2*" => typeof(float2*),
					_ => typeof(void*)
				};
			}

			return type;
		}

		public string GetArgumentTypeString(string typeName)
		{
			string typeIdentifier = RemoveTypeModifiers(typeName).Split(' ').LastOrDefault()?.Trim() ?? "object";
			bool isPointer = typeIdentifier.EndsWith("*");
			if (isPointer)
			{
				typeIdentifier = typeIdentifier.TrimEnd('*').Trim();
			}

			string typeString = typeIdentifier switch
			{
				"int" => "int",
				"float" => "float",
				"double" => "double",
				"char" => "char",
				"bool" => "bool",
				"void" => "void",
				"byte" => "byte",
				"float2" => "Vector2",
				_ => typeIdentifier
			};

			if (isPointer)
			{
				typeString += "*";
			}

			return typeString;
		}

		public Dictionary<string, Type> GetArguments(string? kernelCode = null, bool silent = false)
		{
			kernelCode ??= this.KernelCode;
			if (string.IsNullOrEmpty(kernelCode))
			{
				if (!silent)
				{
					CudaService.Log("Kernel code is empty", this.KernelName ?? "N/A", 1);
				}
				return [];
			}

			Dictionary<string, Type> arguments = [];

			int index = kernelCode.IndexOf("__global__ void");
			if (index == -1)
			{
				if (!silent)
				{
					CudaService.Log($"'__global__ void' not found", this.KernelName ?? "N/A", 1);
				}
				return [];
			}

			index = kernelCode.IndexOf("(", index);
			if (index == -1)
			{
				if (!silent)
				{
					CudaService.Log($"'(' not found", this.KernelName ?? "N/A", 1);
				}
				return [];
			}

			int endIndex = kernelCode.IndexOf(")", index);
			if (endIndex == -1)
			{
				if (!silent)
				{
					CudaService.Log($"')' not found", this.KernelName ?? "N/A", 1);
				}
				return [];
			}

			string[] args = kernelCode.Substring(index + 1, endIndex - index - 1).Split(',').Select(x => x.Trim()).ToArray();

			// Get loaded kernels function args
			for (int i = 0; i < args.Length; i++)
			{
				string name = args[i].Split(' ').LastOrDefault() ?? "N/A";
				string typeName = args[i].Replace(name, "").Trim();
				Type type = this.GetArgumentType(typeName);

				// Add to dictionary
				arguments.Add(name, type);
			}

			return arguments;
		}


		// Merge args for execution
		public object[] MergeArgumentsAudio(CUdeviceptr inputPointer, CUdeviceptr outputPointer, int sampleRate = 44100, int channels = 2, int bitdepth = 32, Dictionary<string, object>? namedArguments = null)
		{
			// Get kernel argument definitions
			Dictionary<string, Type> args = this.GetArguments(null, false);

			// Create array for kernel arguments
			object[] kernelArgs = new object[args.Count];
			int pointersCount = 0;

			// Integrate invariables if name fits (contains)
			for (int i = 0; i < kernelArgs.Length; i++)
			{
				string name = args.ElementAt(i).Key;
				Type type = args.ElementAt(i).Value;
				if (pointersCount == 0 && type == typeof(IntPtr))
				{
					kernelArgs[i] = inputPointer;
					pointersCount++;
					CudaService.Log($"In-pointer: <{inputPointer}>", "", 1);
				}
				else if (pointersCount == 1 && type == typeof(IntPtr))
				{
					kernelArgs[i] = outputPointer;
					pointersCount++;
					CudaService.Log($"Out-pointer: <{outputPointer}>", "", 1);
				}
				else if (name.Contains("sample") && type == typeof(int))
				{
					CudaService.Log($"SampleRate: [{sampleRate}]", "", 1);
				}
				else if (name.Contains("chan") && type == typeof(int))
				{
					kernelArgs[i] = channels;
					CudaService.Log($"Channels: [{channels}]", "", 1);
				}
				else if (name.Contains("bit") && type == typeof(int))
				{
					kernelArgs[i] = bitdepth;
					CudaService.Log($"Bits: [{bitdepth}]", "", 1);
				}
				else
				{
					// Check if argument is in arguments array
					if (namedArguments != null && namedArguments.Count > 0)
					{
						for (int j = 0; j < namedArguments.Count; j++)
						{
							if (name.Equals(args.ElementAt(j).Key, StringComparison.CurrentCultureIgnoreCase))
							{
								if (namedArguments.TryGetValue(name, out object? value))
								{
									kernelArgs[i] = value;
									CudaService.Log($"Named argument: {name} = {value}", "", 1);
									break;
								}
								else
								{
									CudaService.Log($"Named argument '{name}' not found in provided arguments", "", 1);
									kernelArgs[i] = 0;
								}
							}
						}
					}

					// If not found, set to 0
					if (kernelArgs[i] == null)
					{
						kernelArgs[i] = 0;
					}
				}
			}

			return kernelArgs;
		}

		public object[] MergeArgumentsImage(CUdeviceptr inputPointer, CUdeviceptr outputPointer, int width, int height, int channels, int bitdepth, object[] arguments, bool silent = false)
		{
			// Get kernel argument definitions
			Dictionary<string, Type> args = this.GetArguments(null, silent);

			// Create array for kernel arguments
			object[] kernelArgs = new object[args.Count];

			int pointersCount = 0;
			// Integrate invariables if name fits (contains)
			for (int i = 0; i < kernelArgs.Length; i++)
			{
				string name = args.ElementAt(i).Key;
				Type type = args.ElementAt(i).Value;

				if (pointersCount == 0 && type == typeof(IntPtr))
				{
					kernelArgs[i] = inputPointer;
					pointersCount++;

					if (!silent)
					{
						CudaService.Log($"In-pointer: <{inputPointer}>", "", 1);
					}
				}
				else if (pointersCount == 1 && type == typeof(IntPtr))
				{
					kernelArgs[i] = outputPointer;
					pointersCount++;

					if (!silent)
					{
						CudaService.Log($"Out-pointer: <{outputPointer}>", "", 1);
					}
				}
				else if (name.Contains("width") && type == typeof(int))
				{
					kernelArgs[i] = width;

					if (!silent)
					{
						CudaService.Log($"Width: [{width}]", "", 1);
					}
				}
				else if (name.Contains("height") && type == typeof(int))
				{
					kernelArgs[i] = height;

					if (!silent)
					{
						CudaService.Log($"Height: [{height}]", "", 1);
					}
				}
				else if (name.Contains("chan") && type == typeof(int))
				{
					kernelArgs[i] = channels;

					if (!silent)
					{
						CudaService.Log($"Channels: [{channels}]", "", 1);
					}
				}
				else if (name.Contains("bit") && type == typeof(int))
				{
					kernelArgs[i] = bitdepth;

					if (!silent)
					{
						CudaService.Log($"Bits: [{bitdepth}]", "", 1);
					}
				}
				else
				{
					// Check if argument is in arguments array
					for (int j = 0; j < arguments.Length; j++)
					{
						if (name == args.ElementAt(j).Key)
						{
							kernelArgs[i] = arguments[j];
							break;
						}
					}

					// If not found, set to 0
					if (kernelArgs[i] == null)
					{
						kernelArgs[i] = 0;
					}
				}
			}

			// DEBUG LOG
			//CudaService.Log("Kernel arguments: " + string.Join(", ", kernelArgs.Select(x => x.ToString())), "", 1);

			// Return kernel arguments
			return kernelArgs;
		}




	}
}
