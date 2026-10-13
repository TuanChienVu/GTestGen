<?php

// GTestGen V2 ER-trace harness: result returned for a no-op step, or for any step in soft mode.

declare(strict_types=1);

namespace Sylius\Behat\ErTrace;

use Behat\Behat\Tester\Result\StepResult;

final class PassedStepResult implements StepResult
{
    public function isPassed(): bool
    {
        return true;
    }

    public function getResultCode(): int
    {
        return self::PASSED;
    }
}
