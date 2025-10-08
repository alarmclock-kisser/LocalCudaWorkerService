using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Shared
{
	public class AppConfiguration
	{
		public string? ApplicationName { get; set; } = "Local Cuda Worker Service WebApp";
		
		public string? LocalApiUrl { get; set; } = null;
		public int LocalApiPort { get; set; } = 0;
		public string? AutoRegisterDeviceName { get; set; } = null;

		public int MaxLogEntries { get; set; } = 4096;
		public double RefreshIntervalSeconds { get; set; } = 10.0;

		public string? ErrorMessage { get; set; } = null;



		public AppConfiguration()
		{
			// Empty ctor
		}



	}
}
