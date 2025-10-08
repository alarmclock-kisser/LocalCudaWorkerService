using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Shared
{
	public class ApiConfiguration
	{
		public string ApplicationName { get; set; } = string.Empty;

		public string? LocalServerIp { get; set; } = null;
		public string LocalServerUrl { get; set; } = string.Empty;
		public int LocalServerPort { get; set; } = 0;
		public bool UseHttps { get; set; } = false;

		public string? ExternalServerAddress { get; set; } = null;
		public bool SuccessfullyConnectedToExternalServer { get; set; } = false;
		public bool RegisteredAtExternalServer { get; set; } = false;

		public int MaxUploadSizeMb { get; set; } = 0;
		public int DefaultDeviceIndex { get; set; } = -1;
		public string? DefaultDeviceName { get; set; } = null;

		public Dictionary<string, string> AdditionalProperties { get; set; } = [];

		public string? ErrorMessage { get; set; } = null;


		public ApiConfiguration()
		{
			// Empty ctor
		}



	}
}
