import { useState } from "react";
import { useDeleteGroup } from "@/api/groups";
import type { ExperimentGroup } from "@/api/groups";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";

export default function DeleteGroupDialog({
  group,
  experimentCount,
}: {
  group: ExperimentGroup;
  experimentCount: number;
}) {
  const [open, setOpen] = useState(false);
  const deleteMutation = useDeleteGroup();

  function handleDelete(deleteExperiments: boolean) {
    deleteMutation.mutate(
      { id: group.id, deleteExperiments },
      { onSuccess: () => setOpen(false) },
    );
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <button className="text-xs text-muted-foreground hover:text-destructive disabled:opacity-50">
          Usuń grupę
        </button>
      </DialogTrigger>

      <DialogContent>
        <DialogHeader>
          <DialogTitle>Usuń grupę „{group.name}"</DialogTitle>
        </DialogHeader>

        <div className="space-y-4 mt-2">
          {experimentCount > 0 ? (
            <p className="text-sm text-muted-foreground">
              Grupa zawiera <span className="font-medium text-foreground">{experimentCount}</span> eksperymentów.
              Co zrobić z eksperymentami?
            </p>
          ) : (
            <p className="text-sm text-muted-foreground">Grupa jest pusta</p>
          )}

          <div className="flex flex-col gap-2">
            {experimentCount > 0 && (
              <Button
                variant="destructive"
                onClick={() => handleDelete(true)}
                disabled={deleteMutation.isPending}
              >
                Usuń grupę i {experimentCount} eksperymentów
              </Button>
            )}

            <Button
              variant="outline"
              onClick={() => handleDelete(false)}
              disabled={deleteMutation.isPending}
            >
              {experimentCount > 0
                ? "Usuń tylko grupę"
                : "Usuń grupę"}
            </Button>

            <Button variant="ghost" onClick={() => setOpen(false)}>
              Anuluj
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
