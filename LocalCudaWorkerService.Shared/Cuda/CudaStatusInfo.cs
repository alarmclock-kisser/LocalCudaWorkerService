using LocalCudaWorkerService.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Shared.Cuda
{
	public class CudaStatusInfo
	{
		public int DeviceId { get; set; } = -1;
		public string DeviceName { get; set; } = string.Empty;
		public bool Initialized { get; set; } = false;

		public CudaUsageInfo UsageInfo { get; set; } = new();

		public CudaStatusInfo()
		{
			// Empty constructor
		}

		public CudaStatusInfo(CudaService? service)
		{
			if (service == null)
			{
				return;
			}

			this.DeviceId = service.Index;
			this.DeviceName = service.SelectedDevice;
			this.Initialized = service.Initialized;
			this.UsageInfo = new CudaUsageInfo(service.Register);
		}






	}
}
