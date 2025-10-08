using Microsoft.OpenApi.Models;
using Swashbuckle.AspNetCore.SwaggerGen;

public class PlainTextRequestBodyFilter : IOperationFilter
{
	public void Apply(OpenApiOperation operation, OperationFilterContext context)
	{
		if (operation.RequestBody != null && !operation.RequestBody.Content.ContainsKey("text/plain"))
		{
			operation.RequestBody.Content["text/plain"] = new OpenApiMediaType
			{
				Schema = new OpenApiSchema { Type = "string" }
			};
		}
	}
}