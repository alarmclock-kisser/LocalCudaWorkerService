using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Shared.Core
{
	public class AudioObjDto
	{
		public Guid Id { get; set; } = Guid.Empty;
		public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
		public string Name { get; set; } = string.Empty;

		public IEnumerable<float[]> Chunks { get; set; } = [];



	}
}
