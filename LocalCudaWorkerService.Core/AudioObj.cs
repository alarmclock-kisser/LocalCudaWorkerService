using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Core
{
	public class AudioObj
	{
		public Guid Id { get; set; } = Guid.Empty;
		public string Name { get; set; } = string.Empty;
		public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

		public float[] AudioData { get; set; } = [];
		public long Length { get; set; } = 0;

		public int SampleRate { get; set; } = 0;
		public int Channels { get; set; } = 0;
		public int BitDepth { get; set; } = 0;

		public List<float[]> Chunks { get; set; } = [];

		public bool IsProcessing { get; set; } = false;

		public IntPtr Pointer { get; set; } = IntPtr.Zero;
		public string Form { get; set; } = "f";
		public bool OnHost { get; set; } = false;
		public bool OnDevice { get; set; } = false;
		public double StretchFactor { get; set; } = 1.0;

		public Dictionary<string, double> Metrics { get; set; } = [];
		public double? this[string metric]
		{
			get
			{
				if (this.Metrics.ContainsKey(metric))
				{
					return this.Metrics[metric];
				}
				return null;
			}
			set
			{
				if (value == null)
				{
					if (this.Metrics.ContainsKey(metric))
					{
						this.Metrics.Remove(metric);
					}
				}
				else
				{
					this.Metrics[metric] = value.Value;
				}
			}
		}


		public AudioObj()
		{
			// Empty constructor
		}




	}
}
