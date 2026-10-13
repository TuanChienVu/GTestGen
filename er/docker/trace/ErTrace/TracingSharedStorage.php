<?php

// GTestGen V2 ER-trace harness: decorates sylius.behat.shared_storage to record which keys each step writes.

declare(strict_types=1);

namespace Sylius\Behat\ErTrace;

use Sylius\Behat\Service\SharedStorageInterface;

final class TracingSharedStorage implements SharedStorageInterface
{
    public function __construct(private readonly SharedStorageInterface $inner)
    {
    }

    public function get(string $key)
    {
        return $this->inner->get($key);
    }

    public function has(string $key): bool
    {
        return $this->inner->has($key);
    }

    public function set(string $key, $resource): void
    {
        Recorder::event(['kind' => 'storage', 'op' => 'set', 'key' => $key, 'type' => get_debug_type($resource)]);
        $this->inner->set($key, $resource);
    }

    public function remove(string $key): void
    {
        Recorder::event(['kind' => 'storage', 'op' => 'remove', 'key' => $key]);
        $this->inner->remove($key);
    }

    public function getLatestResource()
    {
        return $this->inner->getLatestResource();
    }

    public function setClipboard(array $clipboard)
    {
        $keys = array_map('strval', array_keys($clipboard));
        sort($keys);
        Recorder::event(['kind' => 'storage', 'op' => 'clipboard', 'keys' => $keys]);

        return $this->inner->setClipboard($clipboard);
    }
}
