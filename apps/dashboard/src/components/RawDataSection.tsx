import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import EpochTable from "@/components/EpochTable";

export default function RawDataSection({ experimentId }: { experimentId: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Dane surowe</CardTitle>
      </CardHeader>

      <CardContent>
        <EpochTable experimentId={experimentId} />
      </CardContent>
    </Card>
  );
}
