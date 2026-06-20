FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/dashboard/package.json ./apps/dashboard/

RUN npm install -g pnpm && pnpm install --frozen-lockfile --filter dashboard

COPY apps/dashboard ./apps/dashboard
RUN pnpm --filter dashboard build

FROM node:20-alpine
RUN npm install -g serve
COPY --from=builder /app/apps/dashboard/dist /app/dist
CMD ["serve", "-s", "/app/dist", "-l", "80"]
