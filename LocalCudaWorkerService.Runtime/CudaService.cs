using LocalCudaWorkerService.Core;
using LocalCudaWorkerService.Runtime;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.Design;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions; // added
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Runtime
{
	public class CudaService
	{
		// Objects
		public CudaRegister? Register;
		public CudaFourier? Fourier;
		public CudaCompiler? Compiler;
		public CudaExecutioner? Executioner;

		// Fields
		private Thread? _gpuThread;
		private readonly BlockingCollection<Delegate> _gpuQueue = [];
		private volatile bool gpuRunning;
		private readonly object ctxInitLock = new();
		private readonly SemaphoreSlim gpuLock = new(1, 1);
		private readonly object initLock = new(); // NEU

		private PrimaryContext? CTX;
		private CUdevice? DEV;

		public CudaKernel? Kernel => this.Compiler?.Kernel;
		public string KernelPath => this.Compiler?.KernelPath ?? string.Empty;



		// Attributes
		public string RuntimePath { get; set; } = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "LocalCudaWorkerService.Runtime"));
		public bool Initialized => this.Register != null && this.Fourier != null && this.Compiler != null && this.Executioner != null && this.CTX != null && this.DEV != null;
		public int Index { get; private set; } = -1;
		public Dictionary<CUdevice, string> Devices { get; private set; } = [];

		public BindingList<string> DeviceEntries = [];
		public string SelectedDevice => this.Index >= 0 && this.Index < this.DeviceEntries.Count ? this.Devices.Values.ElementAt(this.Index) : string.Empty;
		public long AllocatedMemory => this.Register?.TotalAllocated ?? 0;
		public int RegisteredMemoryObjects => this.Register?.RegisteredMemoryObjects ?? 0;
		public int MaxThreads = 0;
		public int ThreadsActive => this.Register?.ThreadsActive ?? 0;
		public int ThreadsIdle => this.Register?.ThreadsIdle ?? 0;

		public static BindingList<string> LogEntries { get; set; } = [];
		public static int MaxLogEntries { get; set; } = 1024;
		public static string LogFilePath = string.Empty;
		public static event EventHandler? LogEntryAdded;
		public static bool AggregateSameEntries { get; set; } = false;


		// Constructor
		public CudaService(int index = -1, string device = "RTX", bool logToFile = true)
		{
			this.Devices = this.GetDevices();
			this.DeviceEntries = new BindingList<string>(this.Devices.Values
						.Select((name, idx) => $"[{idx}] {name}")
						.OrderBy(name => name)
						.ToList());

			// Verify runtime path
			if (!Directory.Exists(this.RuntimePath))
			{
				this.RuntimePath = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory));
			}

			// Create / overwrite log file
			if (logToFile)
			{
				LogFilePath = Path.Combine(this.RuntimePath, "LocalCudaWorkerService.log");

				try
				{
					if (File.Exists(LogFilePath))
					{
						File.Delete(LogFilePath);
					}

					using var logFile = File.CreateText(LogFilePath);
					logFile.WriteLine("LocalCudaWorkerService Runtime Log");
					logFile.WriteLine($"Initialized at: {DateTime.Now}");
					logFile.WriteLine(new string('-', 32));
				}
				catch (Exception ex)
				{
					Console.WriteLine($"Failed to create log file: {ex.Message}", "Error", 1);
				}
			}

			// Initialize
			if (index >= 0)
			{
				this.Initialize(index);
			}
			else if (!string.IsNullOrEmpty(device))
			{
				this.Initialize(device);
			}
		}

		// Method: Dispose
		public void Dispose()
		{
			this._gpuQueue.CompleteAdding();
			if (this._gpuThread != null && this._gpuThread.IsAlive)
			{
				this._gpuThread.Join();
			}

			this.Index = -1;

			this.Devices = this.GetDevices();
			this.DeviceEntries = new BindingList<string>(this.Devices.Values
						.Select((name, idx) => $"[{idx}] {name}")
						.OrderBy(name => name)
						.ToList());

			this.CTX?.Dispose();
			this.CTX = null;
			this.DEV = null;

			this.Register?.Dispose();
			this.Register = null;
			this.Fourier?.Dispose();
			this.Fourier = null;
			this.Compiler?.Dispose();
			this.Compiler = null;
			this.Executioner?.Dispose();
			this.Executioner = null;
		}


		// Thread privates
		private void StartGpuThread()
		{
			if (this.gpuRunning)
			{
				return;
			}

			this.gpuRunning = true;
			this._gpuThread = new Thread(() =>
			{
				try
				{
					// Kontext einmalig auf Worker-Thread binden
					this.EnsureContextCurrent();

					foreach (var work in this._gpuQueue.GetConsumingEnumerable())
					{
						switch (work)
						{
							case Action a:
								a();
								break;
							case Func<Task> f:
								// Falls doch async zurück (selten nötig) -> blockend ausführen
								f().GetAwaiter().GetResult();
								break;
							default:
								break;
						}
					}
				}
				finally
				{
					this.gpuRunning = false;
				}
			})
			{
				IsBackground = true,
				Name = "CudaWorkerThread"
			};
			this._gpuThread.Start();
		}

		private Task<T> EnqueueGpu<T>(Func<T> fn)
		{
			var tcs = new TaskCompletionSource<T>(TaskCreationOptions.RunContinuationsAsynchronously);
			this._gpuQueue.Add(new Action(() =>
			{
				try
				{
					// Kontext redundanterweise sichern (falls zukünftige Änderungen)
					this.EnsureContextCurrent();
					var result = fn();
					tcs.SetResult(result);
				}
				catch (Exception ex)
				{
					tcs.SetException(ex);
				}
			}));
			return tcs.Task;
		}

		private Task EnqueueGpu(Action action)
		{
			var tcs = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
			this._gpuQueue.Add(new Action(() =>
			{
				try
				{
					this.EnsureContextCurrent();
					action();
					tcs.SetResult();
				}
				catch (Exception ex)
				{
					tcs.SetException(ex);
				}
			}));
			return tcs.Task;
		}


		// Method: Log (static)
		public static string Log(string message = "", string inner = "", int indent = 0, string? invoker = null, bool addTimeStamp = true)
		{
			if (string.IsNullOrEmpty(invoker))
			{
				// Get the calling class name
				var stackTrace = new StackTrace();
				var frame = stackTrace.GetFrame(1);
				invoker = frame?.GetMethod()?.DeclaringType?.Name ?? "Unknown";
			}

			// PadRight / cut off invoker to 12 characters
			invoker = invoker.PadRight(12).Substring(0, 12);

			// Time stamp as HH:mm:ss.fff (24h)
			string timeStamp = DateTime.Now.ToString("HH:mm:ss.fff");

			// Indentation
			string indentString = new(' ', indent * 2);

			string logMessage = $"[{invoker}{(addTimeStamp ? @" @" + timeStamp : "")}] {indentString}{message} {(string.IsNullOrEmpty(inner) ? "" : $"({inner})")}";

			Console.WriteLine(logMessage);
			LogEntries.Add(logMessage);

			if (LogEntries.Count > MaxLogEntries)
			{
				LogEntries.RemoveAt(0);
			}

			LogEntryAdded?.Invoke(null, EventArgs.Empty);

			// Write to log file
			try
			{
				using var logFile = new StreamWriter(LogFilePath, true);
				logFile.WriteLine(logMessage);
			}
			catch (Exception ex)
			{
				Console.WriteLine($"Failed to write to log file: {ex.Message}", "Error", 1);
			}

			return logMessage;
		}

		public string? GetKernelName(string? code = null)
		{
			if (!string.IsNullOrWhiteSpace(code))
			{
				// Use verbatim string to avoid escape issues.
				var regex = new Regex(@"__global__\s+(?:__launch_bounds__\s*\([^)]*\)\s*)*void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", RegexOptions.Multiline);
				var m = regex.Match(code);
				if (m.Success && m.Groups.Count > 1)
				{
					return m.Groups[1].Value.Trim();
				}

				var lines = code.Split(['\r', '\n'], StringSplitOptions.RemoveEmptyEntries);
				foreach (var line in lines)
				{
					var trimmed = line.Trim();
					if (trimmed.StartsWith("__global__ void"))
					{
						var parts = trimmed.Split([' ', '(', '\t'], StringSplitOptions.RemoveEmptyEntries);
						if (parts.Length >= 3)
						{
							return parts[2];
						}
					}
				}
			}

			if (this.Compiler?.Kernel?.KernelName is not null)
			{
				return this.Compiler.Kernel.KernelName;
			}

			return null;
		}

		public IEnumerable<string> GetKernelCuFilePaths(string nameFilter = "")
		{
			if (this.Compiler == null)
			{
				return [];
			}

			// Get Kernels .cu file paths
			if (string.IsNullOrEmpty(nameFilter))
			{
				return this.Compiler.SourceFiles;
			}
			else
			{
				return this.Compiler.SourceFiles
					.Where(f => Path.GetFileNameWithoutExtension(f).Contains(nameFilter, StringComparison.OrdinalIgnoreCase))
					.ToList();
			}
		}

		public static string Log(Exception exception, string? description = null, int indent = 0, string? invoker = null, bool addTimeStamp = true)
		{
			if (string.IsNullOrEmpty(invoker))
			{
				// Get the calling class name
				var stackTrace = new StackTrace();
				var frame = stackTrace.GetFrame(1);
				invoker = frame?.GetMethod()?.DeclaringType?.Name ?? "Unknown";
			}

			// Time stamp as HH:mm:ss.fff (24h)
			string timeStamp = DateTime.Now.ToString("HH:mm:ss.fff");

			// Indentation
			string indentString = new(' ', indent * 2);

			// Create folded message
			string message = string.Empty;
			if (!string.IsNullOrEmpty(description))
			{
				message += $"{(description.Replace(":", "").Trim())}: ";
			}

			message += exception.Message;
			Exception? inner = exception.InnerException;
			int innerCount = 0;
			while (inner != null)
			{
				message += $" ({inner.Message}";
				inner = inner.InnerException;
				innerCount++;
			}
			if (innerCount > 0)
			{
				message += new string(')', innerCount);
			}

			string logMessage = $"[{invoker}{(addTimeStamp ? @" @" + timeStamp : "")}] {indentString}{message}";

			Console.WriteLine(logMessage);
			LogEntries.Add(logMessage);

			if (LogEntries.Count > MaxLogEntries)
			{
				LogEntries.RemoveAt(0);
			}

			LogEntryAdded?.Invoke(null, EventArgs.Empty);

			// Write to log file
			try
			{
				using var logFile = new StreamWriter(LogFilePath, true);
				logFile.WriteLine(logMessage);
			}
			catch (Exception ex)
			{
				Console.WriteLine($"Failed to write to log file: {ex.Message}", "Error", 1);
			}

			return logMessage;
		}


		// Methods: Initialize
		public void Initialize(int index = 0)
		{
			lock (this.initLock)
			{
				if (this.Initialized && this.Index == index)
				{
					Log($"CUDA bereits initialisiert (Device Index {index}).", "", 0, "CUDA-Service", true);
					return;
				}

				if (this.Initialized)
				{
					this.Dispose();
				}

				if (index < 0 || index >= this.Devices.Count)
				{
					throw new ArgumentOutOfRangeException(nameof(index), $"Ungültiger Device-Index {index}. Verfügbar: 0 .. {this.Devices.Count - 1}");
				}

				try
				{
					this.Index = index;
					this.DEV = this.Devices.Keys.ElementAt(index);

					// Context anlegen (hier noch im aufrufenden Thread)
					this.CTX = new PrimaryContext(this.DEV.Value);

					// Worker-Thread starten (bindet den Context dort)
					this.StartGpuThread();

					// Konstruktion aller GPU-Objekte AUF dem GPU-Thread erzwingen
					var buildTask = this.EnqueueGpu(() =>
					{
						this.EnsureContextCurrent();
						this.MaxThreads = this.CTX!.GetDeviceInfo().MaxThreadsPerMultiProcessor;

						this.Register = new CudaRegister(this.CTX);
						this.Fourier = new CudaFourier(this.CTX, this.Register);
						this.Compiler = new CudaCompiler(this.CTX);
						this.Executioner = new CudaExecutioner(this.CTX, this.Register, this.Fourier, this.Compiler);

						Log($"Initialized CUDA service for device: {this.SelectedDevice} (Index: {this.Index})", "", 0, "CUDA-Service", true);
					});

					buildTask.GetAwaiter().GetResult();
				}
				catch (Exception ex)
				{
					Log(ex, "Failed to initialize CUDA service", 0, "CUDA-Service", true);
					this.Dispose();
				}
			}
		}

		public void Initialize(string device = "NVIDIA")
		{
			int index = this.Devices.Values
				.Select((name, idx) => new { Name = name, Index = idx })
				.FirstOrDefault(x => x.Name.Contains(device, StringComparison.OrdinalIgnoreCase))?.Index ?? -1;

			this.Initialize(index);
		}


		// Method: Devices enumeration 
		internal Dictionary<CUdevice, string> GetDevices(bool silent = true)
		{
			var devices = new Dictionary<CUdevice, string>();

			try
			{
				int deviceCount = CudaContext.GetDeviceCount();

				for (int i = 0; i < deviceCount; i++)
				{
					CUdevice device = new(i);
					string deviceName = CudaContext.GetDeviceName(i);
					devices.Add(device, deviceName);
				}

				if (!silent)
				{
					Log($"Found {deviceCount} CUDA devices.", "", 0, "CUDA-Service", true);
				}
			}
			catch (Exception ex)
			{
				Log(ex, "Failed to enumerate CUDA devices", 0, "CUDA-Service", true);
			}

			return devices;
		}



		// Accessors: Kernel
		public IEnumerable<string> GetUncompiledKernelBuildLogs()
		{
			if (this.Compiler == null)
			{
				return [];
			}

			// Compare cu and ptx files, list every cu filepath that does not have a corresponding ptx file (in
			string[] cuFiles = this.Compiler.SourceFiles.ToArray();
			string[] ptxFiles = this.Compiler.CompiledFiles.ToArray();
			var uncompiled = cuFiles
				.Where(cu => !ptxFiles.Any(ptx => Path.GetFileNameWithoutExtension(ptx).Equals(Path.GetFileNameWithoutExtension(cu), StringComparison.OrdinalIgnoreCase)))
				.ToList();

			string logPath = Path.Combine(this.Compiler.KernelPath, "Logs");
			if (!Directory.Exists(logPath))
			{
				Directory.CreateDirectory(logPath);
			}


			// Read every build log for uncompiled kernels as text
			string[] buildLogFiles = uncompiled
				.Select(cu => Path.Combine(logPath, Path.GetFileNameWithoutExtension(cu) + ".log"))
				.Where(log => File.Exists(log))
				.ToArray();

			var buildLogs = buildLogFiles.SelectMany(logFile =>
				{
					try
					{
						return File.ReadAllLines(logFile);
					}
					catch (Exception ex)
					{
						Log(ex, $"Failed to read build log file: {logFile}", 0, "CUDA-Service", true);
						return [];
					}
				})
				.ToList();

			return buildLogs;
		}

		public IEnumerable<string> GetAvailableKernels(string? filter = null)
		{
			if (this.Compiler == null)
			{
				return [];
			}

			return this.Compiler.SourceFiles.Select(f => Path.GetFileNameWithoutExtension(f))
				.Where(name => string.IsNullOrEmpty(filter) || name.Contains(filter, StringComparison.OrdinalIgnoreCase))
				.OrderBy(name => name);
		}

		public Version? GetComputeCapability(int? deviceIndex = null)
		{
			if (deviceIndex == null)
			{
				deviceIndex = this.Index;
			}
			if (deviceIndex < 0 || deviceIndex >= this.Devices.Count)
			{
				return null;
			}
			try
			{
				var props = CudaContext.GetDeviceInfo(deviceIndex.Value);
				return new Version(props.ComputeCapability.Major, props.ComputeCapability.Minor);
			}
			catch (Exception ex)
			{
				Log(ex, "Failed to get compute capability", 0, "CUDA-Service", true);
				return null;
			}
		}

		public string? GetLatestKernel(IEnumerable<string>? fromKernels = null)
		{
			fromKernels ??= this.Compiler?.SourceFiles ?? [];
			List<string> kernelFiles = [];
			foreach (var kernel in fromKernels ?? [])
			{
				if (!File.Exists(kernel))
				{
					kernelFiles.Add(Path.Combine(this.Compiler?.KernelPath ?? "", "CU", kernel.Replace(".cu", "") + ".cu"));
				}
			}

			var fileInfos = kernelFiles
				.Select(f => new FileInfo(f))
				.Where(fi => fi.Exists)
				.OrderByDescending(fi => fi.LastWriteTime)
				.ToList();

			if (fileInfos.Count == 0)
			{
				Console.WriteLine("No valid kernel files found.", "Error", 1);
				return null;
			}

			var latestFile = Path.GetFileNameWithoutExtension(fileInfos.First().FullName);
			return latestFile;
		}

		public CudaKernel? LoadKernel(string kernelName)
		{
			if (this.Compiler == null)
			{
				return null;
			}

			var kernel = this.Compiler.LoadKernel(kernelName);
			if (kernel == null)
			{
				string? kernelFile = this.Compiler.SourceFiles.FirstOrDefault(f => Path.GetFileNameWithoutExtension(f).Equals(kernelName, StringComparison.OrdinalIgnoreCase));
				if (string.IsNullOrEmpty(kernelFile))
				{
					Console.WriteLine($"Kernel '{kernelName}' not found in source files.", "Error", 1);
					return null;
				}

				string? ptxFile = this.Compiler.CompileKernel(kernelFile);
				if (string.IsNullOrEmpty(ptxFile))
				{
					Console.WriteLine($"Kernel '{kernelName}' could not be compiled.", "Error", 1);
					return null;
				}

				kernel = this.Compiler.LoadKernel(kernelName);
				if (kernel == null)
				{
					Console.WriteLine($"Kernel '{kernelName}' could not be loaded from compiled PTX file.", "Error", 1);
					return null;
				}
			}

			if (kernel == null)
			{
				Console.WriteLine($"Kernel '{kernelName}' could not be loaded.", "Error", 1);
				return null;
			}

			return this.Kernel;
		}

		public string? CompileKernel(string kernelNameOrFile)
		{
			if (this.Compiler == null)
			{
				return null;
			}

			if (File.Exists(kernelNameOrFile))
			{
				return this.Compiler.CompileKernel(kernelNameOrFile);
			}
			else
			{
				kernelNameOrFile = this.Compiler.KernelPath + "\\CU\\" + kernelNameOrFile + ".cu";
				if (File.Exists(kernelNameOrFile))
				{
					return this.Compiler.CompileKernel(kernelNameOrFile);
				}
				else
				{
					Console.WriteLine($"Kernel file '{kernelNameOrFile}' does not exist.", "Error", 1);
					return null;
				}
			}

		}

		public Dictionary<string, Type> GetKernelArguments(string? kernelName = null)
		{
			// Get kernel argument names and types
			if (this.Compiler == null)
			{
				return [];
			}

			kernelName ??= this.Kernel?.KernelName;
			if (string.IsNullOrEmpty(kernelName))
			{
				return [];
			}

			var cuPath = this.Compiler.SourceFiles.FirstOrDefault(f => Path.GetFileNameWithoutExtension(f).Equals(kernelName, StringComparison.OrdinalIgnoreCase));
			if (string.IsNullOrEmpty(cuPath) || !File.Exists(cuPath))
			{
				return [];
			}

			var args = this.Compiler.GetArguments(cuPath);
			return args;
		}

		public string? CreateCuFile(string code = "")
		{
			string kernelName = this.GetKernelName(code) ?? this.Compiler?.Kernel?.KernelName ?? $"kernel_{DateTime.Now:yyyyMMdd_HHmmss}";
			if (string.IsNullOrWhiteSpace(kernelName))
			{
				return null;
			}
			if (this.Compiler == null)
			{
				return null;
			}
			var cuPath = System.IO.Path.Combine(this.KernelPath, "CU", kernelName + ".cu");
			try
			{
				System.IO.File.WriteAllText(cuPath, code);
			}
			catch (Exception ex)
			{
				Log(ex, $"Failed to create CU file: {cuPath}", 0, "CUDA-Service", true);
				return null;
			}
			return cuPath;
		}

		// Accessors
		public AudioObj MoveAudio(AudioObj obj, int chunkSize = 16384, float overlap = 0.5f, bool keep = false)
		{
			if (this.Register == null)
			{
				return obj;
			}

			Stopwatch sw = Stopwatch.StartNew();
			if (obj.OnHost && obj.AudioData.LongLength > 0)
			{
				obj.IsProcessing = true;

				// Move -> Device
				var chunks = obj.Chunks;
				if (chunks == null || !chunks.Any())
				{
					sw.Stop();
					return obj;
				}

				obj["chunk"] = sw.Elapsed.TotalMilliseconds;
				sw.Restart();

				var mem = this.Register.PushChunks(chunks);
				if (mem == null || mem.IndexPointer == IntPtr.Zero)
				{
					sw.Stop();
					return obj;
				}

				obj["push"] = sw.Elapsed.TotalMilliseconds;
				obj.Pointer = mem.IndexPointer;
			}
			else if (obj.OnDevice && obj.Pointer != IntPtr.Zero)
			{
				obj.IsProcessing = true;

				// Move -> Host
				var chunks = this.Register.PullChunks<float>(obj.Pointer, keep);
				if (chunks == null || chunks.Count <= 0)
				{
					sw.Stop();
					return obj;
				}

				obj["pull"] = sw.Elapsed.TotalMilliseconds;
				sw.Restart();

				obj.Chunks = chunks;
				obj["aggregate"] = sw.Elapsed.TotalMilliseconds;
			}

			sw.Stop();
			obj.IsProcessing = false;
			return obj;
		}

		public async Task<AudioObj> MoveAudioAsync(AudioObj obj, int chunkSize = 4096, float overlap = 0.5f)
		{
			if (this.Executioner == null || this.Register == null)
			{
				return obj;
			}

			if (obj.OnHost)
			{
				var chunks = obj.Chunks;
				if (chunks == null || chunks.Count <= 0)
				{
					return obj;
				}

				var mem = await this.Register.PushChunksAsync<float>(chunks);
				if (mem == null || mem.IndexPointer == IntPtr.Zero)
				{
					return obj;
				}

				obj.Pointer = mem.IndexPointer;
			}
			else if (obj.OnDevice)
			{
				var chunks = await this.Register.PullChunksAsync<float>(obj.Pointer);
				if (chunks == null || chunks.Count <= 0)
				{
					return obj;
				}

				obj.Chunks = chunks;
				this.Register.FreeMemory(obj.Pointer);
			}

			return obj;
		}

		public AudioObj FourierTransform(AudioObj obj, int chunkSize = 16384, float overlap = 0.5f, bool keep = false, bool autoPull = false, bool autoNormalize = false, bool asyncFourier = false)
		{
			if (this.Fourier == null || this.Register == null)
			{
				return obj;
			}

			// Move audio to device if not already there
			if (!obj.OnDevice)
			{
				this.MoveAudio(obj, chunkSize, overlap, keep);
			}
			if (obj.Pointer == IntPtr.Zero)
			{
				return obj;
			}

			Stopwatch sw = Stopwatch.StartNew();

			// Perform Fourier Transform
			IntPtr transformedPointer = IntPtr.Zero;
			if (obj.Form == "f")
			{
				obj.IsProcessing = true;

				transformedPointer = asyncFourier
					? this.FourierTransformAsync(obj, chunkSize, overlap, keep, false, autoPull, autoNormalize).GetAwaiter().GetResult().Pointer
					: this.Fourier.PerformFft(obj.Pointer, keep);

				if (transformedPointer != IntPtr.Zero)
				{
					obj.Pointer = transformedPointer;
					obj.Form = "c";
				}
			}
			else if (obj.Form == "c")
			{
				obj.IsProcessing = true;

				transformedPointer = this.Fourier.PerformIfft(obj.Pointer, keep);
				if (transformedPointer != IntPtr.Zero)
				{
					obj.Pointer = transformedPointer;
					obj.Form = "f";
				}
			}
			else
			{
				sw.Stop();
				return obj;
			}

			sw.Stop();
			obj.IsProcessing = false;

			if (autoPull && obj.Form == "f")
			{
				this.MoveAudio(obj, chunkSize, overlap, keep);
			}

			return obj;
		}

		public async Task<AudioObj> FourierTransformAsync(AudioObj obj, int chunkSize = 16384, float overlap = 0.5f, bool keep = false, bool asMany = false, bool autoPull = false, bool autoNormalize = false)
		{
			if (this.Fourier == null || this.Register == null)
			{
				return obj;
			}

			// Move audio to device if not already there
			if (!obj.OnDevice)
			{
				await this.MoveAudioAsync(obj, chunkSize, overlap);
			}
			if (obj.Pointer == IntPtr.Zero)
			{
				return obj;
			}

			Stopwatch sw = Stopwatch.StartNew();

			// Perform Fourier Transform
			IntPtr transformedPointer;
			if (obj.Form == "f")
			{
				obj.IsProcessing = true;

				if (asMany)
				{
					transformedPointer = await this.Fourier.PerformFftManyAsync(obj.Pointer, keep);
				}
				else
				{
					transformedPointer = await this.Fourier.PerformFftAsync(obj.Pointer, keep);
				}

				obj["fft"] = sw.Elapsed.TotalMilliseconds;
			}
			else if (obj.Form == "c")
			{
				obj.IsProcessing = true;

				if (asMany)
				{
					transformedPointer = await this.Fourier.PerformIfftManyAsync(obj.Pointer, keep);
				}
				else
				{
					transformedPointer = await this.Fourier.PerformIfftAsync(obj.Pointer, keep);
				}

				obj["ifft"] = sw.Elapsed.TotalMilliseconds;
			}
			else
			{
				sw.Stop();
				return obj;
			}

			sw.Stop();
			obj.IsProcessing = false;

			if (transformedPointer == IntPtr.Zero)
			{
				Console.WriteLine("Fourier Transform failed, pointer is null.");
				return obj;
			}

			obj.Pointer = transformedPointer;
			obj.Form = obj.Form == "f" ? "c" : "f";

			if (autoPull && obj.Form == "f")
			{
				await this.MoveAudioAsync(obj, chunkSize, overlap);
			}

			return obj;
		}



		public AudioObj TimeStretch(AudioObj obj, string kernel = "timestretch_complexes01", double factor = 1.0, int chunkSize = 16384, float overlap = 0.5f, bool keep = false, bool autoNormalize = false)
		{
			if (this.Executioner == null || this.Compiler == null || this.Fourier == null || this.Register == null)
			{
				return obj;
			}

			// Move audio to device if not already there
			if (!obj.OnDevice)
			{
				this.MoveAudio(obj, chunkSize, overlap, keep);
			}
			if (obj.Pointer == IntPtr.Zero)
			{
				return obj;
			}

			int overlapSize = (int) (chunkSize * overlap);
			IntPtr result = IntPtr.Zero;

			Stopwatch sw = Stopwatch.StartNew();

			obj.IsProcessing = true;
			result = this.Executioner.ExecuteTimeStretch(obj.Pointer, kernel, factor, chunkSize, overlapSize, obj.SampleRate, obj.Channels, keep);
			obj.IsProcessing = false;

			sw.Stop();
			obj["stretch"] = sw.Elapsed.TotalMilliseconds;

			if (result != IntPtr.Zero)
			{
				obj.Pointer = result;
				obj.StretchFactor = factor;
			}

			if (!obj.OnHost)
			{
				this.MoveAudio(obj, chunkSize, overlap);
			}

			return obj;
		}

		public async Task<AudioObj> TimeStretchAsync(AudioObj obj, string kernel = "timestretch_complexes01", double factor = 1.0, int chunkSize = 16384, float overlap = 0.5f, int maxStreams = 1, bool keep = false, bool asMany = false, bool autoNormalize = false)
		{
			if (this.Executioner == null || this.Compiler == null || this.Fourier == null || this.Register == null)
			{
				return obj;
			}

			// Move audio to device if not already there
			if (!obj.OnDevice)
			{
				await this.MoveAudioAsync(obj, chunkSize, overlap);
			}
			if (obj.Pointer == IntPtr.Zero)
			{
				return obj;
			}

			int overlapSize = (int) (chunkSize * overlap);

			obj.IsProcessing = true;
			IntPtr result = IntPtr.Zero;

			Stopwatch sw = Stopwatch.StartNew();

			if (maxStreams == 1)
			{
				result = await this.Executioner.ExecuteTimeStretchAsync(obj.Pointer, kernel, factor, chunkSize, overlapSize, obj.SampleRate, obj.Channels, asMany, keep);
			}
			else
			{
				result = await this.Executioner.ExecuteTimeStretchInterleavedAsync(obj.Pointer, kernel, factor, chunkSize, overlapSize, obj.SampleRate, obj.Channels, maxStreams, asMany, keep);
			}

			sw.Stop();
			obj["stretch"] = sw.Elapsed.TotalMilliseconds;

			obj.IsProcessing = false;

			if (result != IntPtr.Zero)
			{
				obj.Pointer = result;
				obj.StretchFactor = factor;
			}

			if (!obj.OnHost)
			{
				await this.MoveAudioAsync(obj, chunkSize, overlap);
			}

			return obj;
		}

		public static async Task<string[]?>? GetLog(int maxLines = 0)
		{
			// Return log entries as array from in-repo log file (all if maxLines is 0 or below 0)
			try
			{
				if (string.IsNullOrEmpty(LogFilePath) || !File.Exists(LogFilePath))
				{
					return LogEntries.ToArray();
				}
				var allLines = await File.ReadAllLinesAsync(LogFilePath);
				if (maxLines <= 0 || maxLines >= allLines.Length)
				{
					return allLines;
				}
				else
				{
					return allLines.Skip(allLines.Length - maxLines).ToArray();
				}
			}
			catch (Exception ex)
			{
				Log(ex, "Failed to read log file", 0, "CUDA-Service", true);
				return null;
			}
		}

		public AudioObj ExecuteAudioKernel(AudioObj obj, string kernel, int chunkSize = 16384, float overlap = 0.5f, string[]? argNames = null, object[]? argValues = null, float autoNormalize = 0.0f)
		{
			if (this.Executioner == null || this.Compiler == null || this.Register == null)
			{
				return obj;
			}

			// Move audio to device if not already there
			if (!obj.OnDevice)
			{
				this.MoveAudio(obj, chunkSize, overlap);
			}
			if (obj.Pointer == IntPtr.Zero)
			{
				return obj;
			}

			int overlapSize = (int) (chunkSize * overlap);
			Dictionary<string, object> arguments = [];
			if (argNames != null && argValues != null && argNames.Length == argValues.Length)
			{
				for (int i = 0; i < argNames.Length; i++)
				{
					arguments[argNames[i]] = argValues[i];
				}
			}
			IntPtr result = IntPtr.Zero;
			obj.IsProcessing = true;
			Stopwatch sw = Stopwatch.StartNew();
			result = this.Executioner.ExecuteGenericAudioKernel(obj.Pointer, kernel, chunkSize, overlapSize, obj.SampleRate, obj.Channels, arguments);
			obj.IsProcessing = false;
			sw.Stop();
			obj["kernel"] = sw.Elapsed.TotalMilliseconds;
			if (result != IntPtr.Zero)
			{
				obj.Pointer = result;
			}
			if (!obj.OnHost)
			{
				this.MoveAudio(obj, chunkSize, overlap);
			}

			return obj;
		}

		public async Task<AudioObj> ExecuteAudioKernelAsync(AudioObj obj, string kernel, int chunkSize = 16384, float overlap = 0.5f, string[]? argNames = null, object[]? argValues = null, float autoNormalize = 0.0f)
		{
			if (this.Executioner == null || this.Compiler == null || this.Register == null)
			{
				return obj;
			}

			// Move audio to device if not already there
			if (!obj.OnDevice)
			{
				await this.MoveAudioAsync(obj, chunkSize, overlap);
			}
			if (obj.Pointer == IntPtr.Zero)
			{
				return obj;
			}

			int overlapSize = (int) (chunkSize * overlap);
			Dictionary<string, object> arguments = [];
			if (argNames != null && argValues != null && argNames.Length == argValues.Length)
			{
				for (int i = 0; i < argNames.Length; i++)
				{
					arguments[argNames[i]] = argValues[i];
				}
			}

			IntPtr result = IntPtr.Zero;
			Stopwatch sw = Stopwatch.StartNew();

			obj.IsProcessing = true;
			result = await this.Executioner.ExecuteGenericAudioKernelAsync(obj.Pointer, kernel, chunkSize, overlapSize, obj.SampleRate, obj.Channels, arguments);
			obj.IsProcessing = false;

			sw.Stop();
			obj["kernel"] = sw.Elapsed.TotalMilliseconds;
			if (result != IntPtr.Zero)
			{
				obj.Pointer = result;
			}

			if (!obj.OnHost)
			{
				await this.MoveAudioAsync(obj, chunkSize, overlap);
			}

			return obj;
		}

		public async Task<string?> GetKernelCode(String kernelName)
		{
			if (this.Compiler == null)
			{
				return null;
			}

			var cuPath = this.Compiler.SourceFiles.FirstOrDefault(f => Path.GetFileNameWithoutExtension(f).Equals(kernelName, StringComparison.OrdinalIgnoreCase));
			if (string.IsNullOrEmpty(cuPath) || !File.Exists(cuPath))
			{
				return null;
			}

			try
			{
				return await File.ReadAllTextAsync(cuPath);
			}
			catch (Exception ex)
			{
				Log(ex, $"Failed to read CU file: {cuPath}", 0, "CUDA-Service", true);
				return null;
			}
		}

		public async Task<string?> DeleteKernelAsync(string kernelName)
		{
			if (this.Compiler == null)
			{
				return null;
			}
			var cuPath = this.Compiler.SourceFiles.FirstOrDefault(f => Path.GetFileNameWithoutExtension(f).Equals(kernelName, StringComparison.OrdinalIgnoreCase));
			if (string.IsNullOrEmpty(cuPath) || !File.Exists(cuPath))
			{
				return null;
			}

			try
			{
				await Task.Run(() => File.Delete(cuPath));
				return cuPath;
			}
			catch (Exception ex)
			{
				Log(ex, $"Failed to delete CU file: {cuPath}", 0, "CUDA-Service", true);
				return null;
			}
		}

		public Dictionary<string, string> GetDeviceProperties(int deviceId = -1)
		{
			if (deviceId < 0)
			{
				deviceId = this.Index;
			}

			if (deviceId < 0 || deviceId >= this.Devices.Count)
			{
				return [];
			}

			Dictionary<string, string> properties = [];
			var props = CudaContext.GetDeviceInfo(deviceId);
			properties["Name"] = props.DeviceName;
			properties["TotalGlobalMemory"] = (props.TotalGlobalMemory / (1024 * 1024)).ToString() + " MB";
			properties["SharedMemoryPerBlock"] = (props.SharedMemoryPerBlock / 1024).ToString() + " KB";
			properties["ComputeCapability"] = $"{props.ComputeCapability.Major}.{props.ComputeCapability.Minor}";
			properties["ClockRate"] = (props.ClockRate / 1000).ToString() + " MHz";
			properties["MultiProcessorCount"] = props.MultiProcessorCount.ToString();
			properties["MaxThreadsPerMultiProcessor"] = props.MaxThreadsPerMultiProcessor.ToString();
			properties["MaxThreadsPerBlock"] = props.MaxThreadsPerBlock.ToString();
			properties["MaxBlockDim"] = $"[{props.MaxBlockDim.x}, {props.MaxBlockDim.y}, {props.MaxBlockDim.z}]";
			properties["MaxGridDim"] = $"[{props.MaxGridDim.x}, {props.MaxGridDim.y}, {props.MaxGridDim.z}]";
			properties["TotalConstantMemory"] = (props.TotalConstantMemory / 1024).ToString() + " KB";
			properties["WarpSize"] = props.WarpSize.ToString();
			properties["MemoryBusWidth"] = props.GlobalMemoryBusWidth.ToString() + " bits";
			properties["L2CacheSize"] = (props.L2CacheSize / 1024).ToString() + " KB";
			properties["MaxTexture1D"] = $"[{props.MaximumTexture1DWidth}]";
			properties["MaxTexture2D"] = $"[{props.MaximumTexture2DWidth}, {props.MaximumTexture2DHeight}]";
			properties["MaxTexture3D"] = $"[{props.MaximumTexture3DWidth}, {props.MaximumTexture3DHeight}, {props.MaximumTexture3DDepth}]";
			return properties;
		}


		public async Task<AudioObj> MoveAudioAsyncSafe(AudioObj obj, int chunkSize = 4096, float overlap = 0.5f)
			=> await this.EnqueueGpu(() => this.MoveAudio(obj, chunkSize, overlap));

		public async Task<AudioObj> FourierTransformAsyncSafe(AudioObj obj, int chunkSize = 16384, float overlap = 0.5f, bool keep = false, bool autoPull = false, bool autoNormalize = false, bool asyncFourier = false)
			=> await this.EnqueueGpu(() => this.FourierTransform(obj, chunkSize, overlap, keep, autoPull, autoNormalize, asyncFourier));

		public async Task<AudioObj> TimeStretchAsyncSafe(AudioObj obj, string kernel = "timestretch_complexes01", double factor = 1.0, int chunkSize = 16384, float overlap = 0.5f, int maxStreams = 1, bool keep = false, bool asMany = false, bool autoNormalize = false)
			=> await this.EnqueueGpu(() => this.TimeStretch(obj, kernel, factor, chunkSize, overlap, keep, autoNormalize));

		public async Task<AudioObj> ExecuteAudioKernelAsyncSafe(AudioObj obj, string kernel, int chunkSize = 16384, float overlap = 0.5f, string[]? argNames = null, object[]? argValues = null, float autoNormalize = 0.0f)
			=> await this.EnqueueGpu(() => this.ExecuteAudioKernel(obj, kernel, chunkSize, overlap, argNames, argValues, autoNormalize));




		internal async Task<string?> ExecuteGenericKernelAsyncSafe(string kernelCode, string? inputDataBase64 = null, string? inputDataType = null, string outputDataLength = "0", string? outputDataType = null, Dictionary<string, string>? parameters = null, int workDimension = 1, string? onCudaDevice = "NVIDIA")
		{
			// Try compile code
			var compileResult = this.Compiler?.CompileString(kernelCode);
			if (string.IsNullOrEmpty(compileResult) || compileResult.Contains(' '))
			{
				Log("Kernel compilation failed.", "'" + compileResult + "'", 0, "CUDA-Service", true);
				return compileResult;
			}

			if (!this.Initialized)
			{
				if (!string.IsNullOrEmpty(onCudaDevice))
				{
					this.Initialize(onCudaDevice);
				}
				
				if (!this.Initialized)
				{
					Log("CUDA Service not initialized.", "", 0, "CUDA-Service", true);
					return null;
				}
			}

			if (this.Executioner == null || this.Compiler == null || this.Register == null)
			{
				Console.WriteLine("CL-SER | ExecuteGenericDataKernel #000: OpenCL-Services not initialized.");
				return null;
			}


			// Input-Daten dekodieren (falls vorhanden)
			object[]? typedData = null;
			if (!string.IsNullOrWhiteSpace(inputDataBase64) && !string.IsNullOrWhiteSpace(inputDataType))
			{
				switch (inputDataType.ToLowerInvariant())
				{
					case "double":
						typedData = (await ConvertStringToTypeAsync<double>(inputDataBase64)).Cast<object>().ToArray();
						break;
					case "float":
					case "single":
						typedData = (await ConvertStringToTypeAsync<float>(inputDataBase64)).Cast<object>().ToArray();
						break;
					case "int":
					case "int32":
						typedData = (await ConvertStringToTypeAsync<int>(inputDataBase64)).Cast<object>().ToArray();
						break;
					case "byte":
					case "uint8":
						typedData = (await ConvertStringToTypeAsync<byte>(inputDataBase64)).Cast<object>().ToArray();
						break;
					default:
						Console.WriteLine("CL-SER | Unsupported inputDataType: " + inputDataType);
						break;
				}
			}

			long outputLength = long.TryParse(outputDataLength, out long len) ? len : 0;
			if (outputLength <= 0)
			{
				outputLength = int.TryParse(outputDataLength, out int intLen) ? intLen : 0;
			}
			if (outputLength <= 0 && typedData != null)
			{
				outputLength = typedData.LongLength;
			}
			if (outputLength <= 0)
			{
				Console.WriteLine("CL-SER | ExecuteGenericDataKernel #005: Invalid output length.");
				return null;
			}

			object[]? data = null;

			// Kernel ausführen je nach Output-Typ
			switch (outputDataType?.ToLowerInvariant())
			{
				case "int":
				case "int32":
					{
						data = (await this.Executioner.ExecuteGenericKernelSingleAsync<int>(kernelCode, typedData, inputDataType, outputLength, parameters, workDimension)).Cast<object>().ToArray();
						break;
					}
				case "double":
					{
						data = (await this.Executioner.ExecuteGenericKernelSingleAsync<double>(kernelCode, typedData, inputDataType, outputLength, parameters, workDimension)).Cast<object>().ToArray();
						break;
					}
				case "float":
				case "single":
					{
						data = (await this.Executioner.ExecuteGenericKernelSingleAsync<float>(kernelCode, typedData, inputDataType, outputLength, parameters, workDimension)).Cast<object>().ToArray();
						break;
					}
				case "byte":
				case "bytes":
				case "uint8":
					{
						data = (await this.Executioner.ExecuteGenericKernelSingleAsync<byte>(kernelCode, typedData, inputDataType, outputLength, parameters, workDimension)).Cast<object>().ToArray();
						break;
					}
				default:
					Console.WriteLine("CL-SER | ExecuteGenericDataKernel #006: Unsupported outputDataType: " + outputDataType);
					return null;
			}

			// 
			var byteData = new byte[data.Length * sizeof(int)];
			Buffer.BlockCopy(data, 0, byteData, 0, byteData.Length);
			return Convert.ToBase64String(byteData);
		}

		// NEU: Wirklich thread-/context-sichere Variante
		public Task<string?> ExecuteGenericKernelSingleAsyncSafe(string kernelCode,
			string? inputDataBase64 = null, string? inputDataType = null, string outputDataLength = "0", string? outputDataType = null,
			Dictionary<string, string>? parameters = null, int workDimension = 1, string? onCudaDevice = "NVIDIA")
		{
			return this.EnqueueGpu(() =>
			{
				this.EnsureContextCurrent();

				// (1) Initialisierung falls nötig
				if (!this.Initialized)
				{
					if (!string.IsNullOrEmpty(onCudaDevice))
					{
						try { this.Initialize(onCudaDevice); }
						catch (Exception ex)
						{
							Log(ex, "Auto-Initialize failed", 0, "CUDA-Service", true);
							return (string?) null;
						}
					}
					if (!this.Initialized)
					{
						Log("CUDA Service not initialized.", "", 0, "CUDA-Service", true);
						return (string?) null;
					}
				}

				if (this.Executioner == null || this.Compiler == null || this.Register == null)
				{
					Console.WriteLine("CUDA-SER | ExecuteGenericKernelSafe #000: Services not ready.");
					return (string?) null;
				}

				// (2) Kompilieren / Laden im GPU-Thread
				var compileResult = this.Compiler.CompileString(kernelCode);
				if (string.IsNullOrEmpty(compileResult) || compileResult.Contains(' '))
				{
					Log("Kernel compilation failed.", "'" + compileResult + "'", 0, "CUDA-Service", true);
					return compileResult;
				}

				// (3) Input dekodieren
				object[]? typedData = null;
				if (!string.IsNullOrWhiteSpace(inputDataBase64) && !string.IsNullOrWhiteSpace(inputDataType))
				{
					try
					{
						switch (inputDataType.ToLowerInvariant())
						{
							case "double":
								typedData = ConvertStringToTypeAsync<double>(inputDataBase64).GetAwaiter().GetResult().Cast<object>().ToArray();
								break;
							case "float":
							case "single":
								typedData = ConvertStringToTypeAsync<float>(inputDataBase64).GetAwaiter().GetResult().Cast<object>().ToArray();
								break;
							case "int":
							case "int32":
								typedData = ConvertStringToTypeAsync<int>(inputDataBase64).GetAwaiter().GetResult().Cast<object>().ToArray();
								break;
							case "byte":
							case "uint8":
								typedData = ConvertStringToTypeAsync<byte>(inputDataBase64).GetAwaiter().GetResult().Cast<object>().ToArray();
								break;
							default:
								Console.WriteLine("CUDA-SER | Unsupported inputDataType: " + inputDataType);
								break;
						}
					}
					catch (Exception ex)
					{
						Log(ex, "Input decode failed", 0, "CUDA-Service", true);
						return (string?) null;
					}
				}

				// (4) Outputlänge bestimmt
				long outputLength = ParsePositiveLength(outputDataLength);
				if (outputLength <= 0 && typedData != null)
				{
					outputLength = typedData.LongLength;
				}
				if (outputLength <= 0)
				{
					Console.WriteLine($"CL-SER | ExecuteGenericDataKernel #005: Invalid output length. Raw='{outputDataLength}'");
					return (string?) null;
				}

				if (string.IsNullOrWhiteSpace(outputDataType))
				{
					Console.WriteLine("CL-SER | ExecuteGenericDataKernel #006: Missing outputDataType.");
					return (string?) null;
				}

				object[]? dataObjects;

				try
				{
					switch (outputDataType.ToLowerInvariant())
					{
						case "int":
						case "int32":
							dataObjects = this.Executioner
								.ExecuteGenericKernelSingle<int>(kernelCode, typedData, inputDataType, outputLength, parameters, workDimension)
								.Cast<object>().ToArray();
							break;
						case "double":
							dataObjects = this.Executioner
								.ExecuteGenericKernelSingle<double>(kernelCode, typedData, inputDataType, outputLength, parameters, workDimension)
								.Cast<object>().ToArray();
							break;
						case "float":
						case "single":
							dataObjects = this.Executioner
								.ExecuteGenericKernelSingle<float>(kernelCode, typedData, inputDataType, outputLength, parameters, workDimension)
								.Cast<object>().ToArray();
							break;
						case "byte":
						case "bytes":
						case "uint8":
							dataObjects = this.Executioner
								.ExecuteGenericKernelSingle<byte>(kernelCode, typedData, inputDataType, outputLength, parameters, workDimension)
								.Cast<object>().ToArray();
							break;
						default:
							Console.WriteLine("CL-SER | ExecuteGenericDataKernel #007: Unsupported outputDataType: " + outputDataType);
							return (string?) null;
					}
				}
				catch (Exception ex)
				{
					Log(ex, "Kernel execution failed", 0, "CUDA-Service", true);
					return (string?) null;
				}

				if (dataObjects == null || dataObjects.Length == 0)
				{
					return (string?) null;
				}

				// (5) In Base64 serialisieren mit richtiger Elementgröße
				int elemSize = GetElementSize(outputDataType);
				byte[] raw = new byte[dataObjects.Length * elemSize];

				// Boxed Array -> konkretes Array für BlockCopy
				switch (outputDataType.ToLowerInvariant())
				{
					case "int":
					case "int32":
						Buffer.BlockCopy(dataObjects.Cast<int>().ToArray(), 0, raw, 0, raw.Length);
						break;
					case "float":
					case "single":
						Buffer.BlockCopy(dataObjects.Cast<float>().ToArray(), 0, raw, 0, raw.Length);
						break;
					case "double":
						Buffer.BlockCopy(dataObjects.Cast<double>().ToArray(), 0, raw, 0, raw.Length);
						break;
					case "byte":
					case "bytes":
					case "uint8":
						raw = dataObjects.Cast<byte>().ToArray();
						break;
				}

				return Convert.ToBase64String(raw);
			});
		}

		public async Task<string[]?> ExecuteGenericKernelBatchAsyncSafe(string kernelCode,
			string[]? inputDataBase64 = null, string? inputDataType = null, string outputDataLength = "0", string? outputDataStride = "1", string? outputDataType = null,
			Dictionary<string, string>? parameters = null, int workDimension = 1, string? onCudaDevice = "NVIDIA")
		{
			int stride = (int) ParsePositiveLength(outputDataStride);
			if (stride <= 0)
			{
				stride = 1;
			}

			List<string> results = [];
			for (int i = 0; i < stride; i++)
			{
				string? inputData = null;
				if (inputDataBase64 != null && i < inputDataBase64.Length)
				{
					inputData = inputDataBase64[i];
				}

				var res = await this.ExecuteGenericKernelSingleAsyncSafe(kernelCode, inputData, inputDataType, outputDataLength, outputDataType, parameters, workDimension, onCudaDevice);
				
				results.Add(res ?? "");
			}

			return results.ToArray();
		}



		public async Task<float2[]?> ExecuteFftAsyncSafe(float[] floats, bool keepResultBuffer = false)
		{
			if (this.Register == null || this.Fourier == null)
			{
				return null;
			}

			// Enqueue on GPU
			this.EnsureContextCurrent();

			// Push data
			var mem = await this.Register.PushDataAsync(floats);
			if (mem == null || mem.IndexPointer == IntPtr.Zero)
			{
				return null;
			}

			// Perform FFT
			var resultPtr = await this.Fourier.PerformFftAsync(mem.IndexPointer, false);
			if (resultPtr == IntPtr.Zero)
			{
				this.Register.FreeMemory(mem.IndexPointer);
				return null;
			}

			// Pull result
			var result = this.Register.PullData<float2>(resultPtr, keepResultBuffer);

			return result;
		}

		public async Task<float[]?> ExecuteIfftAsyncSafe(float2[] complexes, bool keepResultBuffer = false)
		{
			if (this.Register == null || this.Fourier == null)
			{
				return null;
			}

			// Enqueue on GPU
			this.EnsureContextCurrent();

			// Push data
			var mem = await this.Register.PushDataAsync(complexes);
			if (mem == null || mem.IndexPointer == IntPtr.Zero)
			{
				return null;
			}

			// Perform IFFT
			var resultPtr = await this.Fourier.PerformIfftAsync(mem.IndexPointer, false);
			if (resultPtr == IntPtr.Zero)
			{
				this.Register.FreeMemory(mem.IndexPointer);
				return null;
			}

			// Pull result
			var result = this.Register.PullData<float>(resultPtr, keepResultBuffer);

			return result;
		}

		public async Task<IEnumerable<float2[]>?> ExecuteFftBulkAsyncSafe(IEnumerable<float[]> floatChunks, bool keepResultBuffers = false)
		{
			if (this.Register == null || this.Fourier == null)
			{
				return null;
			}
			
			// Enqueue on GPU
			this.EnsureContextCurrent();
			
			// Push data
			var mem = await this.Register.PushChunksAsync(floatChunks);
			if (mem == null || mem.IndexPointer == IntPtr.Zero)
			{
				return null;
			}

			// Perform FFT
			var resultPtr = await this.Fourier.PerformFftAsync(mem.IndexPointer, false);
			if (resultPtr == IntPtr.Zero)
			{
				this.Register.FreeMemory(mem.IndexPointer);
				return null;
			}
			
			// Pull result
			var results = this.Register.PullChunks<float2>(resultPtr, keepResultBuffers);

			return results;
		}

		public async Task<IEnumerable<float[]>?> ExecuteIfftBulkAsyncSafe(IEnumerable<float2[]> complexChunks, bool keepResultBuffers = false)
		{
			if (this.Register == null || this.Fourier == null)
			{
				return null;
			}
			
			// Enqueue on GPU
			this.EnsureContextCurrent();
			
			// Push data
			var mem = await this.Register.PushChunksAsync(complexChunks);
			if (mem == null || mem.IndexPointer == IntPtr.Zero)
			{
				return null;
			}
			
			// Perform IFFT
			var resultPtr = await this.Fourier.PerformIfftAsync(mem.IndexPointer, false);
			if (resultPtr == IntPtr.Zero)
			{
				this.Register.FreeMemory(mem.IndexPointer);
				return null;
			}
			
			// Pull result
			var results = this.Register.PullChunks<float>(resultPtr, keepResultBuffers);
			
			return results;
		}





		// Statics
		public static async Task<T[]> ConvertStringToTypeAsync<T>(string? base64Data, int parallelThresholdChars = 16_000_000, int? maxDegreeOfParallelism = null, bool ignoreRemainderBytes = true) where T : unmanaged
		{
			if (string.IsNullOrWhiteSpace(base64Data))
			{
				return [];
			}

			try
			{
				// 1) Base64 Decode (synchron, CPU-bound) – optional auslagern
				// Bei extrem großen Strings im Hintergrund-Thread decodieren
				byte[] raw = base64Data.Length > parallelThresholdChars
					? await Task.Run(() => Convert.FromBase64String(base64Data))
					: Convert.FromBase64String(base64Data);

				if (typeof(T) == typeof(byte))
				{
					return (T[]) (object) raw; // Direkt zurück (zero-copy)
				}

				int typeSize = System.Runtime.InteropServices.Marshal.SizeOf<T>();
				if (raw.Length < typeSize)
				{
					return [];
				}

				int elementCount = raw.Length / typeSize;
				int usableBytes = elementCount * typeSize;

				if (!ignoreRemainderBytes && usableBytes != raw.Length)
				{
					throw new InvalidOperationException($"Rohdatenlänge ({raw.Length}) nicht durch Elementgröße ({typeSize}) teilbar.");
				}

				// Klein? -> Singlethread + Array.Copy + Buffer.BlockCopy
				if (base64Data.Length < parallelThresholdChars)
				{
					byte[] usableRaw = new byte[usableBytes];
					Array.Copy(raw, 0, usableRaw, 0, usableBytes);
					T[] result = new T[elementCount];
					Buffer.BlockCopy(usableRaw, 0, result, 0, usableBytes);
					return result;
				}

				// Groß: Parallel kopieren (ohne unsafe)
				T[] resultLarge = new T[elementCount];

				// Partitionierung
				int logical = maxDegreeOfParallelism ?? Environment.ProcessorCount;
				int partitionSizeElements = Math.Max(elementCount / (logical * 4), 1024); // Mindestens 1024 Elemente pro Partition
				int partitionSizeBytes = partitionSizeElements * typeSize;

				var ranges = new List<(int byteStart, int byteLen)>();
				for (int offset = 0; offset < usableBytes; offset += partitionSizeBytes)
				{
					int len = Math.Min(partitionSizeBytes, usableBytes - offset);
					ranges.Add((offset, len));
				}

				var po = new ParallelOptions
				{
					MaxDegreeOfParallelism = logical
				};

				Parallel.ForEach(ranges, po, range =>
				{
					int elements = range.byteLen / typeSize;
					int destIndex = range.byteStart / typeSize;
					Buffer.BlockCopy(raw, range.byteStart, resultLarge, destIndex * typeSize, elements * typeSize);
				});

				return resultLarge;
			}
			catch (FormatException)
			{
				return [];
			}
			catch (Exception ex)
			{
				Console.WriteLine("ConvertStringToTypeAsync error: " + ex.Message);
				return [];
			}
		}

		public static Task<object[]> ConvertStringToTypeAsync(string? base64Data, string typeName)
		{
			return typeName.ToLower() switch
			{
				"byte" or "bytes" => ConvertStringToTypeAsync<byte>(base64Data).ContinueWith(t => t.Result.Cast<object>().ToArray()),
				"sbyte" => ConvertStringToTypeAsync<sbyte>(base64Data).ContinueWith(t => t.Result.Cast<object>().ToArray()),
				"short" or "int16" => ConvertStringToTypeAsync<short>(base64Data).ContinueWith(t => t.Result.Cast<object>().ToArray()),
				"ushort" or "uint16" => ConvertStringToTypeAsync<ushort>(base64Data).ContinueWith(t => t.Result.Cast<object>().ToArray()),
				"int" or "int32" => ConvertStringToTypeAsync<int>(base64Data).ContinueWith(t => t.Result.Cast<object>().ToArray()),
				"uint" or "uint32" => ConvertStringToTypeAsync<uint>(base64Data).ContinueWith(t => t.Result.Cast<object>().ToArray()),
				"long" or "int64" => ConvertStringToTypeAsync<long>(base64Data).ContinueWith(t => t.Result.Cast<object>().ToArray()),
				"ulong" or "uint64" => ConvertStringToTypeAsync<ulong>(base64Data).ContinueWith(t => t.Result.Cast<object>().ToArray()),
				"float" or "single" => ConvertStringToTypeAsync<float>(base64Data).ContinueWith(t => t.Result.Cast<object>().ToArray()),
				"double" => ConvertStringToTypeAsync<double>(base64Data).ContinueWith(t => t.Result.Cast<object>().ToArray()),
				_ => Task.FromResult<object[]>([])
			};
		}

		public static Task<IEnumerable<T[]>> ConvertStringChunksToTypeAsync<T>(IEnumerable<string?> base64Chunks, int parallelThresholdChars = 16_000_000, int? maxDegreeOfParallelism = null, bool ignoreRemainderBytes = true) where T : unmanaged
		{
			var tasks = base64Chunks.Select(b64 => ConvertStringToTypeAsync<T>(b64, parallelThresholdChars, maxDegreeOfParallelism, ignoreRemainderBytes));
			return Task.WhenAll(tasks).ContinueWith(t => (IEnumerable<T[]>) t.Result);
		}

		public static Type GetTypeFromString(string typeName)
		{
			return typeName.ToLower().Trim() switch
			{
				"byte" or "bytes" => typeof(byte),
				"sbyte" => typeof(sbyte),
				"short" or "int16" => typeof(short),
				"ushort" or "uint16" => typeof(ushort),
				"int" or "int32" => typeof(int),
				"uint" or "uint32" => typeof(uint),
				"long" or "int64" => typeof(long),
				"ulong" or "uint64" => typeof(ulong),
				"float" or "single" => typeof(float),
				"double" => typeof(double),
				_ => typeof(object),
			};
		}




		// NEU: Safe Compile Wrapper
		public Task<string?> CompileStringAsyncSafe(string code)
			=> this.EnqueueGpu(() =>
			{
				this.EnsureContextCurrent();
				return this.Compiler?.CompileString(code);
			});

		private static long ParsePositiveLength(string? raw)
		{
			if (string.IsNullOrWhiteSpace(raw))
			{
				return 0;
			}

			string s = raw.Trim();

			// Entferne übliche Formatierungen
			s = s.Replace("_", "").Replace(" ", "").Replace("\u00A0", "");

			// Hex?
			if (s.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
			{
				return long.TryParse(s[2..], System.Globalization.NumberStyles.HexNumber,
					System.Globalization.CultureInfo.InvariantCulture, out long hexVal) && hexVal > 0
					? hexVal : 0;
			}

			if (long.TryParse(s, System.Globalization.NumberStyles.Integer,
				System.Globalization.CultureInfo.InvariantCulture, out long lv) && lv > 0)
			{
				return lv;
			}

			if (double.TryParse(s, System.Globalization.NumberStyles.Float,
				System.Globalization.CultureInfo.InvariantCulture, out double dv) && dv > 0)
			{
				return (long) Math.Floor(dv);
			}

			// Letzter Versuch: nur Ziffern extrahieren
			var digits = new string(s.Where(char.IsDigit).ToArray());
			if (digits.Length > 0 && long.TryParse(digits, out long dv2) && dv2 > 0)
			{
				return dv2;
			}

			return 0;
		}

		private static int GetElementSize(string? typeName)
		{
			return typeName?.ToLowerInvariant() switch
			{
				"byte" or "bytes" or "uint8" => sizeof(byte),
				"int" or "int32" => sizeof(int),
				"float" or "single" => sizeof(float),
				"double" => sizeof(double),
				_ => sizeof(int) // Fallback
			};
		}

		// OPTIONALE VERSTÄRKUNG:
		private bool TryEnsureContext()
		{
			if (this.CTX == null)
			{
				return false;
			}

			try
			{
				this.CTX.SetCurrent();
				return true;
			}
			catch (Exception ex)
			{
				Log(ex, "Context SetCurrent fehlgeschlagen", 0, "CUDA-Service", true);
				return false;
			}
		}

		private void EnsureContextCurrent()
		{
			this.TryEnsureContext();
		}

		public static async Task<string?> ConvertTypeToStringAsync(object[] objects, string? typeName)
		{
			// Try get type if name given, else try infer from objects
			Type? type = null;
			if (!string.IsNullOrWhiteSpace(typeName))
			{
				type = GetTypeFromString(typeName);
			}
			else if (objects.Length > 0 && objects[0] != null)
			{
				type = objects[0].GetType();
			}
			if (type == null)
			{
				return null;
			}

			try
			{
				// Boxed Array -> konkretes Array für BlockCopy
				byte[] raw;
				if (type == typeof(byte))
				{
					raw = (byte[]) (object) objects;
				}
				else if (type == typeof(int))
				{
					raw = new byte[objects.Length * sizeof(int)];
					Buffer.BlockCopy((int[]) (object) objects, 0, raw, 0, raw.Length);
				}
				else if (type == typeof(long))
				{
					raw = new byte[objects.Length * sizeof(long)];
					Buffer.BlockCopy((long[]) (object) objects, 0, raw, 0, raw.Length);
				}
				else if (type == typeof(float))
				{
					raw = new byte[objects.Length * sizeof(float)];
					Buffer.BlockCopy((float[]) (object) objects, 0, raw, 0, raw.Length);
				}
				else if (type == typeof(double))
				{
					raw = new byte[objects.Length * sizeof(double)];
					Buffer.BlockCopy((double[]) (object) objects, 0, raw, 0, raw.Length);
				}
				else
				{
					return null;
				}

				// Base64 Encode
				return await Task.Run(() => Convert.ToBase64String(raw));
			}
			catch (Exception ex)
			{
				Console.WriteLine("ConvertTypeToStringAsync error: " + ex.Message);
				return null;

			}
		}
	}
}
