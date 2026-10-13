<?php

// GTestGen V2 ER-trace harness (not part of Sylius). Process-wide recorder shared by the Behat step wrapper and the app listeners.

declare(strict_types=1);

namespace Sylius\Behat\ErTrace;

final class Recorder
{
    private static int $current = -1;
    private static int $counter = -1;
    /** @var array<int, array<string, mixed>> */
    private static array $steps = [];
    /** @var list<array<string, mixed>> */
    private static array $events = [];
    private static bool $registered = false;
    /** @var array<int, int>|null */
    private static ?array $noop = null;

    public static function enabled(): bool
    {
        return (string) getenv('ER_TRACE_OUT') !== '';
    }

    public static function soft(): bool
    {
        return getenv('ER_SOFT') === '1';
    }

    public static function isNoop(int $i): bool
    {
        if (self::$noop === null) {
            $raw = trim((string) getenv('ER_NOOP'));
            self::$noop = $raw === '' ? [] : array_flip(array_map('intval', explode(',', $raw)));
        }

        return isset(self::$noop[$i]);
    }

    public static function beginStep(string $keywordType, string $text, int $line): int
    {
        self::register();
        self::$current = ++self::$counter;
        self::$steps[self::$current] = ['i' => self::$current, 'type' => $keywordType, 'text' => $text, 'line' => $line];

        return self::$current;
    }

    public static function endStep(int $i, string $status, ?string $error = null): void
    {
        self::$steps[$i]['status'] = $status;
        if ($error !== null) {
            self::$steps[$i]['error'] = mb_substr($error, 0, 400);
        }
        self::$current = -1;
    }

    /** @param array<string, mixed> $event */
    public static function event(array $event): int
    {
        if (!self::enabled()) {
            return -1;
        }
        self::register();
        $event['step'] = self::$current;
        self::$events[] = $event;

        return count(self::$events) - 1;
    }

    public static function annotate(int $idx, string $key, mixed $value): void
    {
        if ($idx >= 0 && isset(self::$events[$idx])) {
            self::$events[$idx][$key] = $value;
        }
    }

    private static function register(): void
    {
        if (self::$registered || !self::enabled()) {
            return;
        }
        self::$registered = true;
        register_shutdown_function(static function (): void {
            file_put_contents(
                (string) getenv('ER_TRACE_OUT'),
                json_encode(['steps' => array_values(self::$steps), 'events' => self::$events],
                    \JSON_UNESCAPED_SLASHES | \JSON_UNESCAPED_UNICODE | \JSON_PARTIAL_OUTPUT_ON_ERROR),
            );
        });
    }
}
