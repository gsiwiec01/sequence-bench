import { useState } from "react";
import type { ExperimentGroup } from "@/api/groups";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";

export default function BulkActionBar({
  count,
  groups,
  canExport,
  onDelete,
  onMove,
  onExport,
  onClear,
}: {
  count: number;
  groups: ExperimentGroup[];
  canExport: boolean;
  onDelete: () => void;
  onMove: (groupId: string) => void;
  onExport: () => void;
  onClear: () => void;
}) {
  const [moveOpen, setMoveOpen] = useState(false);
  const [exporting, setExporting] = useState(false);

  if (count === 0) return null;

  async function handleExport() {
    setExporting(true);
    try {
      await onExport();
    } finally {
      setExporting(false);
    }
  }

  return (
    <>
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 rounded-xl border bg-card shadow-xl px-4 py-2.5 text-sm">
        <span className="font-medium tabular-nums">
          Zaznaczono: <span className="text-primary">{count}</span>
        </span>
        <div className="h-4 w-px bg-border" />
        <Button size="sm" variant="outline" onClick={() => setMoveOpen(true)}>
          Przenieś do grupy
        </Button>
        <Button size="sm" variant="outline" onClick={handleExport} disabled={!canExport || exporting}>
          {exporting ? "Eksportuję..." : "Eksportuj ZIP"}
        </Button>
        <Button size="sm" variant="destructive" onClick={onDelete}>
          Usuń zaznaczone
        </Button>
        <button onClick={onClear} className="text-xs text-muted-foreground hover:text-foreground">
          Odznacz
        </button>
      </div>

      <Dialog open={moveOpen} onOpenChange={setMoveOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Przenieś do grupy</DialogTitle>
          </DialogHeader>

          <div className="mt-2 max-h-72 overflow-y-auto space-y-1.5">
            {groups.length === 0 ? (
              <p className="text-sm text-muted-foreground">Brak grup</p>
            ) : (
              groups.map((g) => (
                <button
                  key={g.id}
                  className="w-full flex items-center justify-between rounded-md border px-3 py-2 text-sm hover:bg-muted transition-colors"
                  onClick={() => { onMove(g.id); setMoveOpen(false); }}
                >
                  <span className="truncate">{g.name}</span>
                  <span className="ml-4 shrink-0 text-xs text-muted-foreground">{g.n_total} eksp.</span>
                </button>
              ))
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
