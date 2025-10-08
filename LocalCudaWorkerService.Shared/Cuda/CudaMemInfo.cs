using LocalCudaWorkerService.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Shared.Cuda
{
	public class CudaMemInfo
	{
		public Guid Id { get; set; } = Guid.Empty;
		public IEnumerable<string> Pointers { get; set; } = [];
		public IEnumerable<string> Lengths { get; set; } = [];
		public string ElementType { get; set; } = string.Empty;
		public string Count { get; set; } = "0";
		public string TotalLength { get; set; } = "0";
		public string TotalBytes { get; set; } = "0";



		public CudaMemInfo()
		{
			// Empty constructor
		}


		public CudaMemInfo(CudaRegister? register, Guid? id)
		{
			if (register == null || !id.HasValue)
			{
				return;
			}

			var mem = register[id.Value];
			if (mem == null)
			{
				return;
			}

			this.Id = mem.Id;
			this.Pointers = mem.Pointers.Select(p => p.ToString("X"));
			this.Lengths = mem.Lengths.Select(l => l.ToString());
			this.ElementType = mem.ElementType.FullName ?? string.Empty;
			this.Count = mem.Count.ToString();
			this.TotalLength = mem.TotalLength.ToString();
			this.TotalBytes = mem.TotalSize.ToString();
		}

		public CudaMemInfo(CudaMem? mem)
		{
			if (mem == null)
			{
				return;
			}

			this.Id = mem.Id;
			this.Pointers = mem.Pointers.Select(p => p.ToString("X"));
			this.Lengths = mem.Lengths.Select(l => l.ToString());
			this.ElementType = mem.ElementType.FullName ?? string.Empty;
			this.Count = mem.Count.ToString();
			this.TotalLength = mem.TotalLength.ToString();
			this.TotalBytes = mem.TotalSize.ToString();
		}

	}
}
