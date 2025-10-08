
using LocalCudaWorkerService.Runtime;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.HttpOverrides;
using Microsoft.OpenApi.Models;

namespace LocalCudaWorkerService.Api
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

			// Get configuration values
            var cudaDeviceId = builder.Configuration.GetValue<int>("DefaultDeviceIndex", -1);
            var cudaDeviceName = builder.Configuration.GetValue<string>("DefaultDeviceName", "");
			var maxUploadSizeMb = builder.Configuration.GetValue<long>("MaxUploadSizeMb", 128) * 1024 * 1024;

			// Add services to the container.
			var cudaService = new CudaService(cudaDeviceId, cudaDeviceName ?? "");
            builder.Services.AddSingleton(cudaService);

			builder.Services.AddEndpointsApiExplorer();
			builder.Services.AddSwaggerGen(o =>
			{
				o.SwaggerDoc("v1", new OpenApiInfo
				{
					Version = "v1",
					Title = "OOCL.Image API",
					Description = "API + WebApp using OpenCL Kernels for image generation etc.",
					TermsOfService = new Uri("https://example.com/terms"),
					Contact = new OpenApiContact { Name = "github: alarmclock-kisser", Email = "marcel.king91299@gmail.com" }
				});
				o.OperationFilter<PlainTextRequestBodyFilter>();
			});

			builder.WebHost.ConfigureKestrel((context, options) =>
			{
				options.Limits.MaxRequestBodySize = maxUploadSizeMb;
				options.Configure(context.Configuration.GetSection("Kestrel"));
			});

			builder.Services.Configure<IISServerOptions>(o => o.MaxRequestBodySize = maxUploadSizeMb);
			builder.Services.Configure<FormOptions>(o => o.MultipartBodyLengthLimit = maxUploadSizeMb);

			builder.Logging.AddConsole().AddDebug();
			builder.Logging.SetMinimumLevel(LogLevel.Debug);

			builder.Services.AddControllers().AddJsonOptions(o =>
			{
				o.JsonSerializerOptions.IncludeFields = true;
			});

			var app = builder.Build();

			app.UseForwardedHeaders(new ForwardedHeadersOptions
			{
				ForwardedHeaders = ForwardedHeaders.XForwardedFor | ForwardedHeaders.XForwardedProto
			});

			// usePathBase ist jetzt false -> kein Prefix mehr
			// app.UsePathBase("/api");

			app.UseHttpsRedirection();

			app.UseSwagger(c => c.RouteTemplate = "swagger/{documentName}/swagger.json");

			app.UseSwaggerUI(c =>
			{
				// Explizit relative Referenz (./) schützt vor Root-Fehlinterpretation auf manchen Clients/Caches
				c.SwaggerEndpoint("./v1/swagger.json", "OOCL.Image API v1");
				c.RoutePrefix = "swagger";
				c.DisplayRequestDuration();
			});

			// Optional: Fallback-Redirect falls Clients noch /swagger/v1/... ohne /api nutzen (IIS Rewrite besser, hier API-Ebene):
			app.MapGet("/swagger/v1/swagger.json", ctx =>
			{
				ctx.Response.Headers.Location = "/api/swagger/v1/swagger.json";
				ctx.Response.StatusCode = StatusCodes.Status302Found;
				return Task.CompletedTask;
			}).ExcludeFromDescription();

			if (app.Environment.IsDevelopment())
			{
				app.UseDeveloperExceptionPage();
			}

			app.MapControllers();;

			app.Run();
		}
    }
}
