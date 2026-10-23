# Stage 1: build frontend
# npm ci downloads with Node's own CA store; 22.23.4 is the first Node 22 that distrusts the
# eMudhra intermediates compromised in October 2026. Pinned, so a cached older node:22-alpine
# cannot build the image.
FROM node:22.23.4-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: serve with nginx
FROM nginx:alpine
# The newest CA store Alpine has, even when the cached base image is older (eMudhra distrust,
# ca-certificates 20261020-r0).
RUN apk upgrade --no-cache ca-certificates ca-certificates-bundle
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Pre-create nginx cache/run directories with correct ownership
# This prevents permission errors when nginx runs with cap_drop: ALL
RUN mkdir -p /var/cache/nginx/client_temp /var/run/nginx && \
    chown -R nginx:nginx /var/cache/nginx /var/run/nginx

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
