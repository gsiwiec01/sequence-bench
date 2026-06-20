import { createFileRoute } from "@tanstack/react-router";
import { useDatasets, useDeleteDataset } from "@/api/datasets";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import DatasetUploadDialog from "@/components/DatasetUpload";

export const Route = createFileRoute("/datasets")({
  component: RouteComponent,
});

function RouteComponent() {
  const { data: datasets = [], isLoading } = useDatasets();
  const deleteMutation = useDeleteDataset();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-3">
        <h1 className="text-2xl font-bold">Datasety</h1>
        <DatasetUploadDialog />
      </div>

      <Card>
        <CardHeader><CardTitle>Dostępne datasety</CardTitle></CardHeader>
        <CardContent>
          {isLoading ? (
            <p className="text-sm text-muted-foreground">Ładowanie...</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Nazwa</TableHead>
                  <TableHead>Typ</TableHead>
                  <TableHead>T</TableHead>
                  <TableHead>d</TableHead>
                  <TableHead>Typ zadania</TableHead>
                  <TableHead></TableHead>
                </TableRow>
              </TableHeader>

              <TableBody>
                {datasets.map((d) => (
                  <TableRow key={d.id}>
                    <TableCell className="font-medium">{d.name}</TableCell>
                    <TableCell>
                      <Badge variant={d.type === "builtin" ? "secondary" : "outline"}>
                        {d.type === "builtin" ? "wbudowany" : "własny"}
                      </Badge>
                    </TableCell>
                    <TableCell>{d.T}</TableCell>
                    <TableCell>{d.input_size}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {d.task_type}
                    </TableCell>
                    <TableCell>
                      {d.type === "custom" && (
                        <Button
                        size="sm"
                            variant="destructive"
                                onClick={() => deleteMutation.mutate(d.id)}>
                          Usuń
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
