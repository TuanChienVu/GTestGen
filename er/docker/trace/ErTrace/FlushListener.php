<?php

// GTestGen V2 ER-trace harness: records the unit of work of every flush (entity class x insert/update/delete, collections).

declare(strict_types=1);

namespace Sylius\Behat\ErTrace;

use Doctrine\ORM\Event\OnFlushEventArgs;

final class FlushListener
{
    public function onFlush(OnFlushEventArgs $args): void
    {
        $uow = $args->getObjectManager()->getUnitOfWork();
        foreach ($uow->getScheduledEntityInsertions() as $entity) {
            Recorder::event(['kind' => 'db', 'op' => 'insert', 'entity' => self::cls($entity)]);
        }
        foreach ($uow->getScheduledEntityUpdates() as $entity) {
            $fields = array_keys($uow->getEntityChangeSet($entity));
            sort($fields);
            Recorder::event(['kind' => 'db', 'op' => 'update', 'entity' => self::cls($entity), 'fields' => $fields]);
        }
        foreach ($uow->getScheduledEntityDeletions() as $entity) {
            Recorder::event(['kind' => 'db', 'op' => 'delete', 'entity' => self::cls($entity)]);
        }
        foreach (['collection_update' => $uow->getScheduledCollectionUpdates(), 'collection_delete' => $uow->getScheduledCollectionDeletions()] as $op => $collections) {
            foreach ($collections as $collection) {
                $mapping = $collection->getMapping();
                Recorder::event(['kind' => 'db', 'op' => $op, 'entity' => self::cls($collection->getOwner()),
                    'field' => is_array($mapping) ? $mapping['fieldName'] : $mapping->fieldName]);
            }
        }
    }

    private static function cls(?object $o): string
    {
        return $o === null ? 'null' : preg_replace('/^.*__CG__\\\\/', '', $o::class);
    }
}
