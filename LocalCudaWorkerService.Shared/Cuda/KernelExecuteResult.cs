using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Shared
{
	public class KernelExecuteResult
	{
		public string KernelName { get; set; } = string.Empty;
		public bool Success { get; set; } = false;
		public string Message { get; set; } = string.Empty;

		public string? OutputPointer { get; set; } = null;
		public string? OutputDataBase64 { get; set; } = null;
		public IEnumerable<string>? OutputDataBase64Chunks { get; set; } = null;
		public string? OutputDataType { get; set; } = null;
		public string OutputDataLength { get; set; } = "0";
		public int? OutputDataStride => this.OutputDataBase64Chunks?.Count() > 1 ? this.OutputDataBase64Chunks.Count() : 1;

		public double? ExecutionTimeMs { get; set; } = null;

		public string? ErrorMessage { get; set; } = null;

		public KernelExecuteResult()
		{
			// Empty default ctor
		}



		public string? GetArrayPreview(int count, bool asOutputType = true)
		{
			if (string.IsNullOrWhiteSpace(this.OutputDataBase64) || string.IsNullOrWhiteSpace(this.OutputDataLength))
			{
				return null;
			}
			try
			{
				long length = long.TryParse(this.OutputDataLength, out long len) ? len : 0;
				length *= 4;
				byte[] data = Convert.FromBase64String(this.OutputDataBase64);
				if (data.LongLength != length)
				{
					return $"[Invalid data length: expected {length}, got {data.Length}]";
				}

				long actualCount = Math.Min(count, length);
				StringBuilder sb = new StringBuilder();
				sb.Append("[");
				for (long i = 0; i < actualCount; i++)
				{
					if (i > 0)
					{
						sb.Append(", ");
					}

					if (asOutputType)
					{
						// Switch based on OutputDataType
						switch (this.OutputDataType?.ToLower())
						{
							case "byte":
							case "sbyte":
								sb.Append(data[i].ToString());
								break;
							case "short":
							case "ushort":
								if (i * 2 + 1 < data.Length)
								{
									short val = BitConverter.ToInt16(data, (int) i * 2);
									sb.Append(val.ToString());
								}
								else
								{
									sb.Append("[...]");
								}
								break;
							case "int":
							case "uint":
								if (i * 4 + 3 < data.Length)
								{
									int val = BitConverter.ToInt32(data, (int) i * 4);
									sb.Append(val.ToString());
								}
								else
								{
									sb.Append("[...]");
								}
								break;
							case "long":
							case "ulong":
								if (i * 8 + 7 < data.Length)
								{
									long val = BitConverter.ToInt64(data, (int) i * 8);
									sb.Append(val.ToString());
								}
								else
								{
									sb.Append("[...]");
								}
								break;
							case "float":
								if (i * 4 + 3 < data.Length)
								{
									float val = BitConverter.ToSingle(data, (int) i * 4);
									sb.Append(val.ToString("G6"));
								}
								else
								{
									sb.Append("[...]");
								}
								break;
							case "double":
								if (i * 8 + 7 < data.Length)
								{
									double val = BitConverter.ToDouble(data, (int) i * 8);
									sb.Append(val.ToString("G6"));
								}
								else
								{
									sb.Append("[...]");
								}
								break;
							default:
								sb.Append($"[Unsupported type: {this.OutputDataType}]");
								i = actualCount; // Exit loop
								break;
						}
					}
					else
					{
						// Raw byte output
						sb.Append(data[i].ToString());
					}
				}
				if (actualCount < length)
				{
					sb.Append(", ...");
				}
				sb.Append("]");
				return sb.ToString();
			}
			catch
			{
				return "[Error decoding data]";
			}
		}

	}
}
