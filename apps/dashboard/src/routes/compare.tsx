import { createFileRoute, Navigate } from "@tanstack/react-router";

export const Route = createFileRoute("/compare")({
  component: () => <Navigate to="/analysis" replace />,
});
