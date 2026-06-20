import {useCallback, useState, DragEvent} from "react";
import {useForm, Controller} from "react-hook-form";
import {zodResolver} from "@hookform/resolvers/zod";
import {z} from "zod";
import {Upload} from "lucide-react";
import {useUploadDataset} from "@/api/datasets";
import {Button} from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {Field, FieldLabel, FieldError} from "@/components/ui/field";
import {Input} from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {cn} from "@/lib/utils";

const schema = z.object({
  name: z.string().min(1, "Nazwa jest wymagana"),
  taskType: z.enum(["classification", "regression", "seq2seq", "language_model"], {
    errorMap: () => ({message: "Wybierz typ zadania"}),
  }),
  nClasses: z.preprocess(
    (v) =>
      v === "" || v === null || v === undefined || (typeof v === "number" && isNaN(v as number))
        ? undefined
        : Number(v),
    z.number().int().min(2, "Minimum 2 klasy").optional(),
  ),
  normalize: z.enum(["zscore", "minmax", "none"]),
  nanStrategy: z.enum(["error", "interpolate", "ffill", "drop"]),
});

type FormValues = z.infer<typeof schema>;

export default function DatasetUploadDialog() {
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [fileError, setFileError] = useState("");
  const mutation = useUploadDataset();

  const {
    register,
    handleSubmit,
    control,
    setValue,
    reset,
    formState: {errors},
  } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {name: "", taskType: "classification", normalize: "zscore", nanStrategy: "error"},
  });

  function handleOpenChange(next: boolean) {
    setOpen(next);
    if (!next) {
      setFile(null);
      setFileError("");
      reset();
    }
  }

  const pickFile = useCallback((f: File) => {
    setFile(f);
    setFileError("");
    setValue("name", f.name.replace(/\.[^.]+$/, ""), {shouldDirty: true});
  }, [setValue]);

  const onDrop = useCallback((e: DragEvent) => {
    e.preventDefault();
    setDragging(false);

    const f = e.dataTransfer.files[0];
    if (f) pickFile(f);
  }, [pickFile]);

  function onSubmit(values: FormValues) {
    if (!file) {
      setFileError("Wybierz plik");
      return;
    }

    const fd = new FormData();
    fd.append("file", file);
    fd.append("name", values.name);
    fd.append("task_type", values.taskType);
    fd.append(
      "config",
      JSON.stringify({
        normalize: values.normalize,
        nan_strategy: values.nanStrategy,
        n_classes: values.nClasses ?? null,
        train_split: 0.7,
        val_split: 0.15,
        test_split: 0.15,
      }),
    );

    mutation.mutate(fd, {
      onSuccess: () => {
        setOpen(false);
        setFile(null);
        reset();
      },
    });
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button>Dodaj dataset</Button>
      </DialogTrigger>

      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Dodaj dataset</DialogTitle>
        </DialogHeader>

        <p className="text-sm text-muted-foreground -mt-1">
          Obsługiwane formaty: .npz, .npy i .csv
        </p>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => document.getElementById("file-input")?.click()}
          className={cn(
            "flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors",
            dragging
              ? "border-primary bg-primary/5"
              : "border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/30",
          )}
        >
          <Upload className="mb-2 h-8 w-8 text-muted-foreground"/>

          {file ? (
            <div className="text-center">
              <p className="font-medium">{file.name}</p>
              <p className="text-sm text-muted-foreground">{(file.size / 1e6).toFixed(1)} MB</p>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Przeciągnij plik lub kliknij</p>
          )}
          <input
            id="file-input"
            type="file"
            accept=".npz,.npy,.csv"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) pickFile(f);
            }}
          />
        </div>

        {fileError && <p className="text-sm text-destructive">{fileError}</p>}
        <form onSubmit={handleSubmit(onSubmit)} className="grid grid-cols-2 gap-4">
          <Field className="col-span-2">
            <FieldLabel>Nazwa datasetu</FieldLabel>
            <Input {...register("name")} placeholder="Mój dataset"/>
            <FieldError errors={[errors.name]}/>
          </Field>

          <Field>
            <FieldLabel>Typ zadania</FieldLabel>
            <Controller
              name="taskType"
              control={control}
              render={({field}) => (
                <Select value={field.value} onValueChange={field.onChange}>
                  <SelectTrigger><SelectValue/></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="classification">Klasyfikacja</SelectItem>
                    <SelectItem value="regression">Regresja</SelectItem>
                    <SelectItem value="seq2seq">Seq2Seq</SelectItem>
                    <SelectItem value="language_model">Language model</SelectItem>
                  </SelectContent>
                </Select>
              )}
            />
            <FieldError errors={[errors.taskType]}/>
          </Field>

          <Field>
            <FieldLabel>
              Liczba klas{" "}
              <span className="font-normal text-muted-foreground">(opcjonalne)</span>
            </FieldLabel>

            <Input
              type="number"
              min={2}
              {...register("nClasses", {valueAsNumber: true})}
              placeholder="np. 10"
            />

            <FieldError errors={[errors.nClasses]}/>
          </Field>

          <Field>
            <FieldLabel>Normalizacja</FieldLabel>
            <Controller
              name="normalize"
              control={control}
              render={({field}) => (
                <Select value={field.value} onValueChange={field.onChange}>
                  <SelectTrigger><SelectValue/></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="zscore">Z-score</SelectItem>
                    <SelectItem value="minmax">Min-Max</SelectItem>
                    <SelectItem value="none">Brak</SelectItem>
                  </SelectContent>
                </Select>
              )}
            />
          </Field>

          <Field>
            <FieldLabel>Obsługa NaN</FieldLabel>
            <Controller
              name="nanStrategy"
              control={control}
              render={({field}) => (
                <Select value={field.value} onValueChange={field.onChange}>
                  <SelectTrigger><SelectValue/></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="error">Błąd (brak NaN)</SelectItem>
                    <SelectItem value="interpolate">Interpolacja</SelectItem>
                    <SelectItem value="ffill">Forward fill</SelectItem>
                    <SelectItem value="drop">Usuń próbki</SelectItem>
                  </SelectContent>
                </Select>
              )}
            />
          </Field>

          {mutation.error && (
            <p className="col-span-2 text-sm text-destructive">{mutation.error.message}</p>
          )}

          <Button
            type="submit"
            disabled={!file || mutation.isPending}
            className="col-span-2 w-full"
          >
            {mutation.isPending ? "Wysyłam..." : "Prześlij dataset"}
          </Button>
        </form>
      </DialogContent>
    </Dialog>
  );
}
