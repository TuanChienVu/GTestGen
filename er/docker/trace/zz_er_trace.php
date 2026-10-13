<?php

// GTestGen V2 ER-trace harness: app services, copied to config/packages/test_cached/ in the trace image only.

declare(strict_types=1);

use Sylius\Behat\ErTrace\FlushListener;
use Sylius\Behat\ErTrace\RequestListener;
use Sylius\Behat\ErTrace\TracingSharedStorage;
use Symfony\Component\DependencyInjection\Loader\Configurator\ContainerConfigurator;

use function Symfony\Component\DependencyInjection\Loader\Configurator\service;

return static function (ContainerConfigurator $container): void {
    $services = $container->services();
    $services->set('er_trace.request_listener', RequestListener::class)
        ->tag('kernel.event_listener', ['event' => 'kernel.request', 'method' => 'onRequest', 'priority' => 4096])
        ->tag('kernel.event_listener', ['event' => 'kernel.response', 'method' => 'onResponse', 'priority' => -4096]);
    $services->set('er_trace.flush_listener', FlushListener::class)
        ->tag('doctrine.event_listener', ['event' => 'onFlush']);
    $services->set('er_trace.shared_storage', TracingSharedStorage::class)
        ->decorate('sylius.behat.shared_storage')
        ->args([service('.inner')]);
};
