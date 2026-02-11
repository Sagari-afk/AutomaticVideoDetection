CREATE DATABASE  IF NOT EXISTS `video_handler`;
USE `video_handler`;

DROP TABLE IF EXISTS `video` cascade ;

CREATE TABLE `video` (
  `id` int NOT NULL AUTO_INCREMENT,
  `video_name` varchar(255) DEFAULT NULL,
  `path` varchar(255) NOT NULL,
  PRIMARY KEY (`id`)
) ;

DROP TABLE IF EXISTS `des_files` cascade;

CREATE TABLE `des_files` (
  `id` int NOT NULL AUTO_INCREMENT,
  `video_id` int DEFAULT NULL,
  `path` varchar(255) DEFAULT NULL,
  `language` varchar(3) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `des_files_video_video_id_idx` (`video_id`),
  CONSTRAINT `des_files_video_video_id_fk` FOREIGN KEY (`video_id`) REFERENCES `video` (`id`)
) ;

DROP TABLE IF EXISTS `detect_files` cascade;

CREATE TABLE `detect_files` (
  `id` int NOT NULL AUTO_INCREMENT,
  `video_id` int DEFAULT NULL,
  `path` varchar(255) DEFAULT NULL,
  `detect_type` varchar(8) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `detect_files_video_video_id_fk_idx` (`video_id`),
  CONSTRAINT `detect_files_video_video_id_fk` FOREIGN KEY (`video_id`) REFERENCES `video` (`id`)
) ;

DROP TABLE IF EXISTS `transition_videos` cascade;

CREATE TABLE `transition_videos` (
  `id` int NOT NULL AUTO_INCREMENT,
  `video_name` varchar(255) NOT NULL,
  `path` varchar(255) NOT NULL,
  `element_paths` text,
  PRIMARY KEY (`id`),
  UNIQUE KEY `video_name` (`video_name`),
  UNIQUE KEY `path_UNIQUE` (`path`)
) ;

DROP TABLE IF EXISTS `translations` cascade;

CREATE TABLE `translations` (
  `id` int NOT NULL AUTO_INCREMENT,
  `sk_translation` varchar(255) NOT NULL,
  `en_translation` varchar(255) NOT NULL,
  `wid_name` varchar(255) NOT NULL,
  `parent` varchar(45) NOT NULL,
  PRIMARY KEY (`id`)
) ;


DROP TABLE IF EXISTS `video_files` cascade;

CREATE TABLE if not exists `video_files` (
  `id` int NOT NULL AUTO_INCREMENT,
  `video` blob,
  `video_name` varchar(255) NOT NULL,
  `path` varchar(255) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `video_name` (`video_name`),
  UNIQUE KEY `path_UNIQUE` (`path`)
) ;

DROP TABLE IF EXISTS `information_files` cascade;

CREATE TABLE `information_files` (
  `id` int NOT NULL AUTO_INCREMENT,
  `file` blob,
  `video_id` int DEFAULT NULL,
  `file_name` varchar(255) NOT NULL,
  `path` varchar(255) NOT NULL,
  `trans_video_id` int DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `path_UNIQUE` (`path`),
  UNIQUE KEY `file_name_UNIQUE` (`file_name`),
  KEY `information_files_transition_videos_id_fk` (`trans_video_id`),
  KEY `information_files_video_files_id_fk` (`video_id`),
  CONSTRAINT `information_files_transition_videos_id_fk` FOREIGN KEY (`trans_video_id`) REFERENCES `transition_videos` (`id`),
  CONSTRAINT `information_files_video_files_id_fk` FOREIGN KEY (`video_id`) REFERENCES `video_files` (`id`)
) ;

