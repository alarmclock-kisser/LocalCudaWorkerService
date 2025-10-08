using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Shared.Cuda
{
	public class KernelExecuteRequest
	{
		public int? DeviceIndex { get; set; } = null;
		public string? DeviceName { get; set; } = null;

		public string? KernelName { get; set; } = null;
		public string? KernelCode { get; set; } = null;

		public IEnumerable<string> ArgumentTypes { get; set; } = [];
		public IEnumerable<string> ArgumentNames { get; set; } = [];
		public IEnumerable<string> ArgumentValues { get; set; } = [];

		public int WorkDimension { get; set; } = 1;

		public string? InputDataBase64 { get; set; } = null;
		public IEnumerable<string>? InputDataBase64Chunks { get; set; } = null;
		public string? InputDataType { get; set; } = null;
		public int InputDataStride => this.InputDataBase64Chunks?.Count() > 1 ? this.InputDataBase64Chunks.Count() : 1;

		public string OutputDataType { get; set; } = "byte";
		public string OutputDataLength { get; set; } = "0";

		public KernelExecuteRequest()
		{
			// Empty ctor
		}



	}
}
