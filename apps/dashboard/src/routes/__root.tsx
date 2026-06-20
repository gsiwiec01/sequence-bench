import { createRootRoute, Outlet, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { request } from "@/api/client";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

interface HealthData {
  status: "ok" | "degraded";
  db: string;
  redis: string;
  gpu: string;
}

function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => request<HealthData>("/health"),
    refetchInterval: 30_000,
    retry: false,
    staleTime: 20_000,
  });
}

function serviceDotClass(value: string): string {
  if (value === "ok") return "bg-emerald-500";
  if (value === "none" || value === "unavailable") return "bg-muted-foreground/40";
  if (value.startsWith("error")) return "bg-destructive";
  return "bg-emerald-500"; // GPU device name = available
}

function serviceLabel(key: string, value: string): string {
  if (value === "ok") return key === "gpu" ? "dostępne" : "ok";
  if (value === "none") return "brak";
  if (value === "unavailable") return "niedostępny";
  if (value.startsWith("error: ")) return value.slice(7, 40);
  return value;
}

function HealthWidget() {
  const { data, isError, isFetching } = useHealth();

  const overallDot = isError
    ? "bg-destructive"
    : !data
      ? "bg-muted-foreground/30 animate-pulse"
      : data.status === "ok"
        ? "bg-emerald-500"
        : "bg-amber-500";

  const overallLabel = isError
    ? "błąd połączenia"
    : !data
      ? "sprawdzam…"
      : data.status === "ok"
        ? "wszystko działa"
        : "problemy";

  const services = [
    { key: "db", label: "PostgreSQL" },
    { key: "redis", label: "Redis" },
    { key: "gpu", label: "GPU" },
  ] as const;

  return (
    <div className="group relative">
      <button
        type="button"
        className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      >
        <span
          className={cn(
            "h-2 w-2 rounded-full transition-colors",
            overallDot,
            isFetching && data && "animate-pulse",
          )}
        />
        <span className="font-mono">API</span>
      </button>

      {/* Dropdown -visible on hover */}
      <div className="absolute right-0 top-full z-50 mt-1.5 hidden w-56 group-hover:block">
        <div className="rounded-lg border bg-card shadow-lg">
          <div className="border-b px-3 py-2.5">
            <div className="flex items-center gap-2">
              <span className={cn("h-2 w-2 rounded-full shrink-0", overallDot)} />
              <span className="text-xs font-medium">{overallLabel}</span>
            </div>
          </div>
          <div className="px-3 py-2 space-y-2">
            {isError ? (
              <p className="text-xs text-destructive py-1">
                Nie można połączyć z API. Sprawdź czy serwer działa.
              </p>
            ) : !data ? (
              <p className="text-xs text-muted-foreground py-1">Ładowanie…</p>
            ) : (
              services.map(({ key, label }) => (
                <div key={key} className="flex items-center justify-between gap-3 text-xs">
                  <span className="text-muted-foreground shrink-0">{label}</span>
                  <div className="flex items-center gap-1.5 min-w-0">
                    <span
                      className={cn(
                        "h-1.5 w-1.5 rounded-full shrink-0",
                        serviceDotClass(data[key]),
                      )}
                    />
                    <span
                      className={cn(
                        "font-mono truncate",
                        data[key] === "ok"
                          ? "text-emerald-600 dark:text-emerald-400"
                          : data[key] === "none" || data[key] === "unavailable"
                            ? "text-muted-foreground"
                            : data[key].startsWith("error")
                              ? "text-destructive"
                              : "text-foreground",
                      )}
                    >
                      {serviceLabel(key, data[key])}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Nav links
// ---------------------------------------------------------------------------

const NAV_LINKS = [
  { to: "/", label: "Dashboard", exact: true },
  { to: "/analysis", label: "Analiza", exact: false },
  { to: "/datasets", label: "Zbiory danych", exact: false },
] as const;

// ---------------------------------------------------------------------------
// Root
// ---------------------------------------------------------------------------

export const Route = createRootRoute({
  component: () => (
    <div className="min-h-screen bg-background font-sans antialiased">
      <nav className="sticky top-0 z-40 border-b bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/80">
        <div className="container mx-auto flex h-13 items-center gap-5 px-6">
          {/* Brand */}
          <Link
            to="/"
            className="flex items-center font-mono text-sm font-semibold tracking-tight text-foreground"
          >
            sequence-bench
          </Link>

          <div className="h-4 w-px bg-border shrink-0" />

          {/* Nav links */}
          <ul className="flex items-center gap-1">
            {NAV_LINKS.map(({ to, label, exact }) => (
              <li key={to}>
                <Link
                  to={to}
                  activeOptions={{ exact }}
                  className="rounded-md px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                  activeProps={{
                    className:
                      "rounded-md px-3 py-1.5 text-sm font-medium bg-muted text-foreground",
                  }}
                >
                  {label}
                </Link>
              </li>
            ))}
          </ul>

          <div className="flex-1" />

          <HealthWidget />
        </div>
      </nav>

      <main className="container mx-auto px-6 py-8">
        <Outlet />
      </main>
    </div>
  ),
});
