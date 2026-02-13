# MuJoCo WASM 3D Viewer
# Serves static files with correct WASM MIME type

FROM nginx:alpine

# Add WASM MIME type inside the types { } block in mime.types
RUN sed -i 's/}/    application\/wasm  wasm;\n}/' /etc/nginx/mime.types

# Custom nginx config with CORS
RUN printf 'server {\n\
    listen 80;\n\
    server_name localhost;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
\n\
    add_header Access-Control-Allow-Origin "*" always;\n\
\n\
    location / {\n\
        try_files $uri $uri/ /index.html;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf

# Copy viewer files
COPY . /usr/share/nginx/html/

EXPOSE 80
