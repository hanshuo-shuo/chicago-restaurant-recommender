CREATE TABLE `events` (
	`id` text PRIMARY KEY NOT NULL,
	`impression_id` text NOT NULL,
	`request_id` text NOT NULL,
	`session_id` text NOT NULL,
	`item_id` text NOT NULL,
	`type` text NOT NULL,
	`position` integer NOT NULL,
	`model` text NOT NULL,
	`model_version` text NOT NULL,
	`timestamp` integer NOT NULL,
	`synthetic` integer DEFAULT 1 NOT NULL,
	`source` text DEFAULT 'portfolio_demo' NOT NULL,
	FOREIGN KEY (`impression_id`) REFERENCES `impressions`(`id`) ON UPDATE no action ON DELETE cascade,
	CONSTRAINT "events_type" CHECK("events"."type" IN ('click','favorite')),
	CONSTRAINT "events_synthetic" CHECK("events"."synthetic"=1),
	CONSTRAINT "events_source" CHECK("events"."source"='portfolio_demo')
);
--> statement-breakpoint
CREATE INDEX `events_time` ON `events` (`timestamp`);--> statement-breakpoint
CREATE INDEX `events_impression` ON `events` (`impression_id`);--> statement-breakpoint
CREATE TABLE `impressions` (
	`id` text PRIMARY KEY NOT NULL,
	`request_id` text NOT NULL,
	`session_id` text NOT NULL,
	`item_id` text NOT NULL,
	`position` integer NOT NULL,
	`model` text NOT NULL,
	`model_version` text NOT NULL,
	`timestamp` integer NOT NULL,
	`synthetic` integer DEFAULT 1 NOT NULL,
	`source` text DEFAULT 'portfolio_demo' NOT NULL,
	CONSTRAINT "impressions_synthetic" CHECK("impressions"."synthetic"=1),
	CONSTRAINT "impressions_source" CHECK("impressions"."source"='portfolio_demo')
);
--> statement-breakpoint
CREATE UNIQUE INDEX `impressions_request_session_item` ON `impressions` (`request_id`,`session_id`,`item_id`);--> statement-breakpoint
CREATE INDEX `impressions_session_time` ON `impressions` (`session_id`,`timestamp`);--> statement-breakpoint
CREATE INDEX `impressions_time` ON `impressions` (`timestamp`);