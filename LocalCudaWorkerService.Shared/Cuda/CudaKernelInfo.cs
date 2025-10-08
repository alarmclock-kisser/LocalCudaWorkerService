using LocalCudaWorkerService.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LocalCudaWorkerService.Shared.Cuda
{
	public class CudaKernelInfo
	{
		public string Name { get; set; } = string.Empty;

		public IEnumerable<string> ArgumentTypes { get; set; } = [];
		public IEnumerable<string> ArgumentNames { get; set; } = [];

		public string InputType { get; set; } = "void*";
		public string ReturnType { get; set; } = "void*";

		public bool SuccessfullyCompiled { get; set; } = false;
		public string CompilationLog { get; set; } = string.Empty;




		public CudaKernelInfo()
		{
			// Parameterless constructor for serialization
		}

		public CudaKernelInfo(CudaCompiler? compiler, string kernelNameOrCode = "", int? index = null)
		{
			if (compiler == null)
			{
				return;
			}

			if (string.IsNullOrEmpty(kernelNameOrCode))
			{
				if (index.HasValue)
				{
					kernelNameOrCode = Path.GetFileNameWithoutExtension(compiler.GetPtxFiles().ElementAtOrDefault(index.Value) ?? "") ?? "";
				}
				else
				{
					return;
				}
			}

			string? code = null;
			var kernelCu = compiler.GetCuFiles().FirstOrDefault(f => string.IsNullOrEmpty(kernelNameOrCode) || Path.GetFileNameWithoutExtension(f).Contains(kernelNameOrCode, StringComparison.OrdinalIgnoreCase));
			var kernelPtx = compiler.GetPtxFiles().FirstOrDefault(f => string.IsNullOrEmpty(kernelNameOrCode) || Path.GetFileNameWithoutExtension(f).Contains(kernelNameOrCode, StringComparison.OrdinalIgnoreCase));
			if (kernelCu == null || kernelPtx == null)
			{
				code = kernelNameOrCode;
			}
			else
			{
				code = System.IO.File.ReadAllText(kernelCu);
				if (string.IsNullOrEmpty(code))
				{
					return;
				}

				this.Name = Path.GetFileNameWithoutExtension(kernelCu);
			}

			var arguments = compiler.GetArguments(code);
			this.ArgumentTypes = arguments.Select(a => a.Value.Name);
			this.ArgumentNames = arguments.Select(a => a.Key);

			this.InputType = arguments.Values.FirstOrDefault(t => t.Name.Contains("*", StringComparison.OrdinalIgnoreCase))?.Name ?? "void*";
			this.ReturnType = arguments.Values.LastOrDefault(t => t.Name.Contains("*", StringComparison.OrdinalIgnoreCase))?.Name ?? "void*";
		}


	}
}
