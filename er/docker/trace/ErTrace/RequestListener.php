<?php

// GTestGen V2 ER-trace harness: records every main request handled by the kernel (ApiClient / KernelBrowser traffic).

declare(strict_types=1);

namespace Sylius\Behat\ErTrace;

use Symfony\Component\HttpKernel\Event\RequestEvent;
use Symfony\Component\HttpKernel\Event\ResponseEvent;

final class RequestListener
{
    private int $pending = -1;

    public function onRequest(RequestEvent $event): void
    {
        if (!$event->isMainRequest()) {
            return;
        }
        $request = $event->getRequest();
        $data = json_decode((string) $request->getContent(), true);
        if (!is_array($data)) {
            $data = $request->request->all();
        }
        $this->pending = Recorder::event([
            'kind' => 'http',
            'method' => $request->getMethod(),
            'path' => $request->getPathInfo(),
            'query' => self::keyPaths($request->query->all()),
            'body_keys' => array_map('strval', array_keys($data)),
            'body_paths' => self::keyPaths($data),
            'files' => array_map('strval', array_keys($request->files->all())),
        ]);
    }

    public function onResponse(ResponseEvent $event): void
    {
        if (!$event->isMainRequest()) {
            return;
        }
        $request = $event->getRequest();
        $route = $request->attributes->get('_route');
        $tokens = [];
        foreach ((array) $request->attributes->get('_route_params', []) as $name => $value) {
            if (is_scalar($value) && stripos((string) $name, 'token') !== false) {
                $tokens[] = (string) $value;  // random per run (e.g. order tokenValue): normalised on the host
            }
        }
        Recorder::annotate($this->pending, 'status', $event->getResponse()->getStatusCode());
        Recorder::annotate($this->pending, 'route', is_string($route) ? $route : null);
        Recorder::annotate($this->pending, 'token_values', $tokens);
        $this->pending = -1;
    }

    /** @return list<string> */
    private static function keyPaths(array $data, string $prefix = ''): array
    {
        $out = [];
        foreach ($data as $key => $value) {
            $path = $prefix . (is_int($key) ? '[]' : ($prefix === '' ? '' : '.') . $key);
            if (is_array($value) && $value !== []) {
                array_push($out, ...self::keyPaths($value, $path));
            } else {
                $out[] = $path;
            }
        }
        $out = array_values(array_unique($out));
        sort($out);

        return $out;
    }
}
