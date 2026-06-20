import { cn } from "@/lib/utils";
import { ChevronDown } from "lucide-react";

export default function Collapsible({
  title,
  summary,
  open,
  onToggle,
  children,
}: {
  title: string;
  summary?: string;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-md border">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center justify-between px-4 py-2.5 text-sm font-medium transition-colors hover:bg-muted/50"
      >
        <span>{title}</span>
        <div className="flex items-center gap-3">
          {!open && summary && (
            <span className="text-xs font-normal text-muted-foreground">{summary}</span>
          )}
          <ChevronDown
            className={cn(
              "h-4 w-4 text-muted-foreground transition-transform duration-200",
              open && "rotate-180",
            )}
          />
        </div>
      </button>
      {open && <div className="border-t px-4 py-4">{children}</div>}
    </div>
  );
}
