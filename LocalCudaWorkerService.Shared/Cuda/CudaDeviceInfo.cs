using LocalCudaWorkerService.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Shared.Cuda
{
	public class CudaDeviceInfo
	{
		public int DeviceId { get; set; } = -1;

		public string DeviceName { get; set; } = string.Empty;
		public string TotalGlobalMemory { get; set; } = string.Empty;
		public string SharedMemoryPerBlock { get; set; } = string.Empty;
		public string ComputeCapability { get; set; } = string.Empty;
		public string ClockRate { get; set; } = string.Empty;
		public string MultiProcessorCount { get; set; } = string.Empty;
		public string MaxThreadsPerMultiProcessor { get; set; } = string.Empty;
		public string MaxThreadsPerBlock { get; set; } = string.Empty;
		public string MaxBlockDim { get; set; } = string.Empty;
		public string MaxGridDim { get; set; } = string.Empty;
		public string TotalConstantMemory { get; set; } = string.Empty;
		public string WarpSize { get; set; } = string.Empty;
		public string MemoryBusWidth { get; set; } = string.Empty;
		public string L2CacheSize { get; set; } = string.Empty;
		public string MaxTexture1D { get; set; } = string.Empty;
		public string MaxTexture2D { get; set; } = string.Empty;
		public string MaxTexture3D { get; set; } = string.Empty;


		public CudaDeviceInfo()
		{
			// Empty ctor
		}

		public CudaDeviceInfo(CudaService? service, int? index)
		{
			if (service == null || !index.HasValue)
			{
				return;
			}

			var props = service.GetDeviceProperties(index.Value);

			this.DeviceId = index.Value;
			this.DeviceName = props.GetValueOrDefault("Name", string.Empty);
			this.TotalGlobalMemory = props.GetValueOrDefault("TotalGlobalMemory", string.Empty);
			this.SharedMemoryPerBlock = props.GetValueOrDefault("SharedMemoryPerBlock", string.Empty);
			this.ComputeCapability = props.GetValueOrDefault("ComputeCapability", string.Empty);
			this.ClockRate = props.GetValueOrDefault("ClockRate", string.Empty);
			this.MultiProcessorCount = props.GetValueOrDefault("MultiProcessorCount", string.Empty);
			this.MaxThreadsPerMultiProcessor = props.GetValueOrDefault("MaxThreadsPerMultiProcessor", string.Empty);
			this.MaxThreadsPerBlock = props.GetValueOrDefault("MaxThreadsPerBlock", string.Empty);
			this.MaxBlockDim = props.GetValueOrDefault("MaxBlockDim", string.Empty);
			this.MaxGridDim = props.GetValueOrDefault("MaxGridDim", string.Empty);
			this.TotalConstantMemory = props.GetValueOrDefault("TotalConstantMemory", string.Empty);
			this.WarpSize = props.GetValueOrDefault("WarpSize", string.Empty);
			this.MemoryBusWidth = props.GetValueOrDefault("MemoryBusWidth", string.Empty);
			this.L2CacheSize = props.GetValueOrDefault("L2CacheSize", string.Empty);
			this.MaxTexture1D = props.GetValueOrDefault("MaxTexture1D", string.Empty);
			this.MaxTexture2D = props.GetValueOrDefault("MaxTexture2D", string.Empty);
			this.MaxTexture3D = props.GetValueOrDefault("MaxTexture3D", string.Empty);
		}

		public override string ToString()
		{
			string nl = Environment.NewLine;
			return $"DeviceId: {this.DeviceId}{nl}" +
				   $"DeviceName: {this.DeviceName}{nl}" +
				   $"TotalGlobalMemory: {this.TotalGlobalMemory}{nl}" +
				   $"SharedMemoryPerBlock: {this.SharedMemoryPerBlock}{nl}" +
				   $"ComputeCapability: {this.ComputeCapability}{nl}" +
				   $"ClockRate: {this.ClockRate}{nl}" +
				   $"MultiProcessorCount: {this.MultiProcessorCount}{nl}" +
				   $"MaxThreadsPerMultiProcessor: {this.MaxThreadsPerMultiProcessor}{nl}" +
				   $"MaxThreadsPerBlock: {this.MaxThreadsPerBlock}{nl}" +
				   $"MaxBlockDim: {this.MaxBlockDim}{nl}" +
				   $"MaxGridDim: {this.MaxGridDim}{nl}" +
				   $"TotalConstantMemory: {this.TotalConstantMemory}{nl}" +
				   $"WarpSize: {this.WarpSize}{nl}" +
				   $"MemoryBusWidth: {this.MemoryBusWidth}{nl}" +
				   $"L2CacheSize: {this.L2CacheSize}{nl}" +
				   $"MaxTexture1D: {this.MaxTexture1D}{nl}" +
				   $"MaxTexture2D: {this.MaxTexture2D}{nl}" +
				   $"MaxTexture3D: {this.MaxTexture3D}";
		}

	}
}
