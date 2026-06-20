import { useState } from "react";
import { useRenameGroup } from "@/api/groups";
import type { ExperimentGroup } from "@/api/groups";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export default function RenameGroupDialog({
  group,
  open,
  onOpenChange,
}: {
  group: ExperimentGroup;
  open: boolean;
  onOpenChange: (v: boolean) => void;
}) {
  const [name, setName] = useState(group.name);
  const renameMutation = useRenameGroup();

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    renameMutation.mutate(
      { id: group.id, name },
      { onSuccess: () => onOpenChange(false) },
    );
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Zmień nazwę grupy</DialogTitle>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4 mt-2">
          <div className="space-y-1.5">
            <Label htmlFor="rename-group">Nowa nazwa</Label>
            <Input
              id="rename-group"
              value={name}
              onChange={(e) => setName(e.target.value)}
              autoFocus
              required
            />
          </div>

          <div className="flex justify-end gap-2">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Anuluj</Button>
            <Button type="submit" disabled={!name.trim() || renameMutation.isPending}>
              Zapisz
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
