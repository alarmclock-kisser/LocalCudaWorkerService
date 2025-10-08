using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Shared.Cuda
{
	public class CuFftRequest
	{
		public string? DeviceName { get; set; } = null;

		public int Size { get; set; } = 0;
		public int Batches { get; set; } = 1;
		public bool? Inverse { get; set; } = null;

		// Data is float[][] or float2[][]
		public IEnumerable<object[]> DataChunks { get; set; } = [];
		public string DataForm { get; set; } = "f";



	}
}
