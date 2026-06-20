export const BASE_URL = import.meta.env.VITE_API_URL ?? "";

export async function request<T>(path: string, {headers, ...options}: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: {"Content-Type": "application/json", ...headers},
    ...options,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({detail: res.statusText}));
    throw new Error((body as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }

  if (res.status === 204) return undefined as T;

  return res.json() as Promise<T>;
}

export async function downloadFile(path: string, filename: string): Promise<void> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({detail: res.statusText}));
    throw new Error((body as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
