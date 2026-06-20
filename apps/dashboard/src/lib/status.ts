export const STATUS_LABEL: Record<string, string> = {
  pending: "oczekuje",
  running: "trwa",
  completed: "ukończone",
  failed: "błąd",
  cancelled: "anulowane",
};

export const STATUS_VARIANT: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  completed: "default",
  running: "secondary",
  pending: "outline",
  failed: "destructive",
  cancelled: "outline",
};
