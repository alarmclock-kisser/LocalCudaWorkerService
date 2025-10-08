using LocalCudaWorkerService.Client;
using LocalCudaWorkerService.Shared;
using LocalCudaWorkerService.WebApp.Components;
using Radzen;
using System; // added for Uri

namespace LocalCudaWorkerService.WebApp
{
	public class Program
	{
		public static void Main(string[] args)
		{
			var builder = WebApplication.CreateBuilder(args);

			// Get configuration
			var localApiUrl = builder.Configuration.GetValue<string>("LocalApiUrl", "https://localhost");
			var localApiPort = builder.Configuration.GetValue<int>("LocalApiPort", 12345);
			var autoRegisterDeviceName = builder.Configuration.GetValue<string>("AutoRegisterDeviceName", string.Empty);

			// Add services to the container.
			var appConfig = new AppConfiguration()
			{
				ApplicationName = builder.Environment.ApplicationName ?? "ASP.NET Blazor (server-sided) WebApp using .NET8",
				LocalApiUrl = localApiUrl,
				LocalApiPort = localApiPort,
				AutoRegisterDeviceName = string.IsNullOrWhiteSpace(autoRegisterDeviceName) ? null : autoRegisterDeviceName
			};
			builder.Services.AddSingleton(appConfig);

			builder.Services.AddRazorPages();
			builder.Services.AddServerSideBlazor();
			builder.Services.AddRadzenComponents();

			// Build safe client base URI (avoid double-port like "https://localhost:32141:32141")
			var baseUrlStr = (appConfig.LocalApiUrl ?? "https://localhost").TrimEnd('/');
			if (!Uri.TryCreate(baseUrlStr, UriKind.Absolute, out var parsedBase))
			{
				throw new InvalidOperationException($"Invalid LocalApiUrl: {baseUrlStr}");
			}

			Uri clientBaseUri;
			if (parsedBase.IsDefaultPort)
			{
				var ub = new UriBuilder(parsedBase) { Port = appConfig.LocalApiPort };
				clientBaseUri = ub.Uri;
			}
			else
			{
				// LocalApiUrl already specifies a port — trust it and ignore LocalApiPort
				clientBaseUri = parsedBase;
			}

			// Register LocalApiClient (constructed from validated URI)
			var apiClient = new LocalApiClient(clientBaseUri.ToString().TrimEnd('/'));
			builder.Services.AddSingleton(apiClient);

			// Optionally register named HttpClient for other consumers (use same base)
			builder.Services.AddHttpClient("LocalApi", client =>
			{
				client.BaseAddress = clientBaseUri;
				client.Timeout = TimeSpan.FromSeconds(10);
			});

			builder.Services.AddSignalR(o => o.MaximumReceiveMessageSize = 1024 * 1024 * 1024);
			builder.Services.AddServerSideBlazor(options =>
			{
				options.DetailedErrors = true;
			}).AddHubOptions(o => { o.MaximumReceiveMessageSize = 1024 * 1024 * 1024; });

			var app = builder.Build();

			// Configure the HTTP request pipeline.
			if (!app.Environment.IsDevelopment())
			{
				app.UseExceptionHandler("/Error");
				app.UseHsts();
			}

			app.UseHttpsRedirection();


			app.UseStaticFiles();
			app.UseRouting();
			app.UseAntiforgery();

			app.MapBlazorHub();
			app.MapFallbackToPage("/_Host");
			app.MapRazorPages();

			app.Run();
		}
	}
}
