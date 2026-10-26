import { sql } from 'drizzle-orm';
import { sqliteTable, text, integer, index, uniqueIndex, check } from 'drizzle-orm/sqlite-core';

export const impressions = sqliteTable('impressions', {
  id: text('id').primaryKey(), requestId: text('request_id').notNull(), sessionId: text('session_id').notNull(),
  itemId: text('item_id').notNull(), position: integer('position').notNull(), model: text('model').notNull(),
  modelVersion: text('model_version').notNull(), timestamp: integer('timestamp').notNull(),
  synthetic: integer('synthetic').notNull().default(1), source: text('source').notNull().default('portfolio_demo'),
}, t => [uniqueIndex('impressions_request_session_item').on(t.requestId,t.sessionId,t.itemId),
  index('impressions_session_time').on(t.sessionId,t.timestamp), index('impressions_time').on(t.timestamp),
  check('impressions_synthetic',sql`${t.synthetic}=1`), check('impressions_source',sql`${t.source}='portfolio_demo'`)]);

export const events = sqliteTable('events', {
  id: text('id').primaryKey(), impressionId: text('impression_id').notNull().references(()=>impressions.id,{onDelete:'cascade'}),
  requestId: text('request_id').notNull(), sessionId: text('session_id').notNull(), itemId: text('item_id').notNull(),
  type: text('type').notNull(), position: integer('position').notNull(), model: text('model').notNull(),
  modelVersion: text('model_version').notNull(), timestamp: integer('timestamp').notNull(),
  synthetic: integer('synthetic').notNull().default(1), source: text('source').notNull().default('portfolio_demo'),
}, t => [index('events_time').on(t.timestamp), index('events_impression').on(t.impressionId),
  check('events_type',sql`${t.type} IN ('click','favorite')`), check('events_synthetic',sql`${t.synthetic}=1`),
  check('events_source',sql`${t.source}='portfolio_demo'`)]);
