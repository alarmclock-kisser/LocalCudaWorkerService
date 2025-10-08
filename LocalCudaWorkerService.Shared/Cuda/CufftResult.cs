using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Shared.Cuda
{
	public class CufftResult
	{
		public IEnumerable<object[]> DataChunks { get; set; } = [];
		public string DataForm { get; set; } = "f";
	}
}
