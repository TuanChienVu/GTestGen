# GTestGen V2 — ER harness image for Sylius @ c096ca3 (arm64).
# Stages: base (PHP + extensions + composer) -> src (Sylius at the pinned SHA)
#         -> lockgen (one-off: resolve composer.lock) | app (install from pinned lock, warm cache).
# Build:   docker compose -f er/docker/compose.yaml build sylius
# Lockgen: docker build -f er/docker/Dockerfile --target lockgen-out --output type=local,dest=er/docker/lock er/docker

ARG PHP_IMAGE=php:8.4-cli@sha256:2b87a5d0c0465ade44b1bbec46e8f63520448a7bd7ec90a284dd8b8f0b07fd53

FROM ${PHP_IMAGE} AS base
ARG PHP_EXT_INSTALLER=2.11.33
ADD --chmod=0755 https://github.com/mlocati/docker-php-extension-installer/releases/download/${PHP_EXT_INSTALLER}/install-php-extensions /usr/local/bin/
RUN apt-get update \
 && apt-get install -y --no-install-recommends git unzip ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && install-php-extensions intl gd exif zip pdo_mysql
COPY --from=composer/composer:2.10.3-bin /composer /usr/local/bin/composer
COPY php.ini /usr/local/etc/php/conf.d/zz-er.ini
ENV COMPOSER_ALLOW_SUPERUSER=1 COMPOSER_NO_INTERACTION=1

FROM base AS src
ARG SYLIUS_SHA=c096ca3fc01545ee25078ba3df914bd2069c7fa6
WORKDIR /app
RUN git init -q . \
 && git remote add origin https://github.com/Sylius/Sylius.git \
 && git fetch -q --depth 1 origin ${SYLIUS_SHA} \
 && git -c advice.detachedHead=false checkout -q FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${SYLIUS_SHA}"

# One-off lock resolution. Global flex applies composer.json extra.symfony.require (^7.4), as Sylius CI does.
FROM src AS lockgen
RUN composer global config --no-plugins allow-plugins.symfony/flex true \
 && composer global require -q --no-progress --no-scripts --no-plugins "symfony/flex:^2.4" \
 && composer update --no-install --no-progress --no-scripts
FROM scratch AS lockgen-out
COPY --from=lockgen /app/composer.lock /composer.lock

FROM src AS app
ARG COMPOSER_LOCK_SHA256
COPY lock/composer.lock /app/composer.lock
RUN echo "${COMPOSER_LOCK_SHA256}  /app/composer.lock" | sha256sum -c - \
 && composer install --no-progress --no-scripts \
 && mkdir -p public/build/admin public/build/shop public/build/app/admin public/build/app/shop \
 && for d in admin shop app/admin app/shop; do echo '{}' > public/build/$d/manifest.json; done
ENV APP_ENV=test_cached APP_DEBUG=0 \
    DATABASE_URL="mysql://root:root@db:3306/sylius_warmup?serverVersion=8.4.0&charset=utf8mb4"
RUN bin/console assets:install public -q \
 && bin/console cache:warmup -q \
 && git diff --quiet HEAD \
 && test -z "$(git status --porcelain --untracked-files=all | grep -v -E '^\?\? public/build/')"
COPY er-inner.sh /usr/local/bin/er-inner
RUN chmod 0755 /usr/local/bin/er-inner

# ER-trace variant (descriptive metric): harness-only tracers, never in overlays. Tracked tree stays == c096ca3;
# .git is removed after that check (runs never need it, and model code should not see it).
FROM app AS trace
COPY trace/ErTrace/ /app/src/Sylius/Behat/ErTrace/
COPY trace/zz_er_trace.php /app/config/packages/test_cached/zz_er_trace.php
RUN rm -rf var/cache/test_cached \
 && bin/console cache:warmup -q \
 && git diff --quiet HEAD \
 && test -z "$(git status --porcelain --untracked-files=all | grep -v -E '^\?\? (public/build/|src/Sylius/Behat/ErTrace/|config/packages/test_cached/zz_er_trace.php)')" \
 && rm -rf /app/.git
