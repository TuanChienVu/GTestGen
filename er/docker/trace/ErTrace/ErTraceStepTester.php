<?php

// GTestGen V2 ER-trace harness: wraps Behat's step tester to mark step boundaries, no-op chosen steps (ER_NOOP) and,
// in soft mode (ER_SOFT=1), keep executing after a failure so every step gets a real result.

declare(strict_types=1);

namespace Sylius\Behat\ErTrace;

use Behat\Behat\Tester\Result\StepResult;
use Behat\Behat\Tester\StepTester;
use Behat\Gherkin\Node\FeatureNode;
use Behat\Gherkin\Node\StepNode;
use Behat\Testwork\Environment\Environment;
use Behat\Testwork\Tester\Result\ExceptionResult;
use Behat\Testwork\Tester\Setup\Setup;
use Behat\Testwork\Tester\Setup\Teardown;

final class ErTraceStepTester implements StepTester
{
    private const NAMES = [0 => 'passed', 10 => 'skipped', 20 => 'pending', 30 => 'undefined', 99 => 'failed'];

    public function __construct(private readonly StepTester $baseTester)
    {
    }

    public function setUp(Environment $env, FeatureNode $feature, StepNode $step, bool $skip): Setup
    {
        return $this->baseTester->setUp($env, $feature, $step, Recorder::soft() ? false : $skip);
    }

    public function test(Environment $env, FeatureNode $feature, StepNode $step, bool $skip): StepResult
    {
        $i = Recorder::beginStep((string) $step->getKeywordType(), (string) $step->getText(), (int) $step->getLine());
        if (Recorder::isNoop($i)) {
            Recorder::endStep($i, 'noop');

            return new PassedStepResult();
        }
        $result = $this->baseTester->test($env, $feature, $step, Recorder::soft() ? false : $skip);
        $error = null;
        if ($result instanceof ExceptionResult && $result->hasException()) {
            $e = $result->getException();
            $error = $e::class . ': ' . $e->getMessage();
        }
        Recorder::endStep($i, self::NAMES[$result->getResultCode()] ?? (string) $result->getResultCode(), $error);

        return Recorder::soft() && !$result->isPassed() ? new PassedStepResult() : $result;
    }

    public function tearDown(Environment $env, FeatureNode $feature, StepNode $step, bool $skip, StepResult $result): Teardown
    {
        return $this->baseTester->tearDown($env, $feature, $step, Recorder::soft() ? false : $skip, $result);
    }
}
