import { createRootRoute, Outlet, Link } from "@tanstack/react-router";

const NAV_LINKS = [
  { to: "/", label: "Dashboard" },
  { to: "/experiments/new", label: "Nowy eksperyment" },
  { to: "/results", label: "Wyniki" },
  { to: "/datasets", label: "Datasety" },
  { to: "/gradients", label: "Gradienty" },
] as const;

export const Route = createRootRoute({
  component: () => (
    <div className="min-h-screen bg-background font-sans antialiased">
      <nav className="border-b bg-card px-6 py-3">
        <ul className="flex items-center gap-6">
          {NAV_LINKS.map(({ to, label }) => (
            <li key={to}>
              <Link
                to={to}
                className="text-sm font-medium text-muted-foreground transition-colors hover:text-foreground [&.active]:font-semibold [&.active]:text-foreground"
                activeProps={{ className: "active" }}
                activeOptions={{ exact: to === "/" }}
              >
                {label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
      <main className="container mx-auto px-6 py-8">
        <Outlet />
      </main>
    </div>
  ),
});
