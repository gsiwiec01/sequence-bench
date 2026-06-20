import { createFileRoute, Navigate } from "@tanstack/react-router";

export const Route = createFileRoute("/results")({
  component: () => <Navigate to="/analysis" replace />,
});
