<?php

// GTestGen V2 ER-trace harness: Behat extension, enabled only through BEHAT_PARAMS by the ER runner.

declare(strict_types=1);

namespace Sylius\Behat\ErTrace;

use Behat\Behat\Tester\ServiceContainer\TesterExtension;
use Behat\Testwork\ServiceContainer\Extension;
use Behat\Testwork\ServiceContainer\ExtensionManager;
use Symfony\Component\Config\Definition\Builder\ArrayNodeDefinition;
use Symfony\Component\DependencyInjection\ContainerBuilder;
use Symfony\Component\DependencyInjection\Definition;
use Symfony\Component\DependencyInjection\Reference;

final class ErTraceExtension implements Extension
{
    public function getConfigKey(): string
    {
        return 'er_trace';
    }

    public function initialize(ExtensionManager $extensionManager): void
    {
    }

    public function configure(ArrayNodeDefinition $builder): void
    {
    }

    public function load(ContainerBuilder $container, array $config): void
    {
        $definition = new Definition(ErTraceStepTester::class, [new Reference(TesterExtension::STEP_TESTER_ID)]);
        $definition->addTag(TesterExtension::STEP_TESTER_WRAPPER_TAG, ['priority' => 0]);
        $container->setDefinition(TesterExtension::STEP_TESTER_WRAPPER_TAG . '.er_trace', $definition);
    }

    public function process(ContainerBuilder $container): void
    {
    }
}
