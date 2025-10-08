using LocalCudaWorkerService.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Shared.Cuda
{
	public class CudaUsageInfo
	{
		public double TotalMemoryMb { get; set; } = 0;
		public double FreeMemoryMb { get; set; } = 0;
		public double UsedMemoryMb { get; set; } = 0;

		public CudaUsageInfo()
		{
			// Empty constructor
		}

		public CudaUsageInfo(CudaRegister? register)
		{
			if (register == null)
			{
				return;
			}

			this.TotalMemoryMb = register.TotalMemory / (1024.0 * 1024.0);
			this.FreeMemoryMb = register.TotalFree / (1024.0 * 1024.0);
			this.UsedMemoryMb = register.TotalAllocated / (1024.0 * 1024.0);
		}

	}
}
