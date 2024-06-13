SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;
drop database if exists `db`;
create database if not exists db;
use db;
show global variables like '%secure%';
set global secure_file_priv = "";


-- ----------------------------
-- Table structure for t_inference_data
-- ----------------------------
DROP TABLE IF EXISTS `t_inference_data`;
CREATE TABLE `t_inference_data`  (
  `Id` int(11) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `Rt_code` int(11) NULL DEFAULT NULL COMMENT '采集数据ID',
  `Inference_time` datetime(0) NULL DEFAULT NULL COMMENT '推理时间',
  `Inference_path` varchar(4000) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '最优结果路径',
  `Inference_score` float(6, 2) NULL DEFAULT NULL COMMENT '推理结果最优路径分数',
  `Inference_root_cause_id` varchar(40) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '最优路径根原因位点',
  `Inference_root_cause_status` tinyint(1) NOT NULL DEFAULT 1 COMMENT '最优路径根原因位点状态 （上升2、下降1）',
  PRIMARY KEY (`Id`) USING BTREE,
  INDEX `key_code`(`Rt_code`, `Inference_time`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 3 CHARACTER SET = latin1 COLLATE = latin1_swedish_ci COMMENT = '在线推理表' ROW_FORMAT = Compact PARTITION BY HASH (`Id`)
PARTITIONS 32
(PARTITION `p0` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p1` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p10` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p11` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p12` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p13` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p14` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p15` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p16` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p17` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p18` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p19` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p2` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p20` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p21` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p22` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p23` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p24` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p25` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p26` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p27` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p28` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p29` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p3` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p30` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p31` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p4` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p5` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p6` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p7` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p8` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p9` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 )
;

-- ----------------------------
-- Records of t_inference_data
-- ----------------------------
LOAD DATA INFILE 'D:\\tmp\\query\\code\\a_dataset\\database\\t_inference_data.csv' 
INTO TABLE t_inference_data
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES;

select * from t_inference_data;


-- ----------------------------
-- Table structure for t_model_data
-- ----------------------------
DROP TABLE IF EXISTS `t_model_data`;
CREATE TABLE `t_model_data`  (
  `Id` int(11) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `Tag_code` varchar(40) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '点位编码',
  `Inference_path` text CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '推理路径',
  `Inference_root_cause_id` varchar(40) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '根源因编码',
  PRIMARY KEY (`Id`) USING BTREE,
  INDEX `key_code1`(`Tag_code`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 5 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '模型表' ROW_FORMAT = Compact;

-- ----------------------------
-- Records of t_model_data
-- ----------------------------
LOAD DATA INFILE 'D:\\tmp\\query\\code\\a_dataset\\database\\t_model_data.csv' 
INTO TABLE t_model_data
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES;

select * from t_model_data;


-- ----------------------------
-- Table structure for t_rt_data
-- ----------------------------
DROP TABLE IF EXISTS `t_rt_data`;
CREATE TABLE `t_rt_data`  (
  `Id` int(11) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `Tag_code` varchar(40) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '点位编码',
  `Result_value` double(10, 3) NULL DEFAULT NULL COMMENT '采集值',
  `Result_time` datetime(0) NULL DEFAULT NULL COMMENT '采集时间',
  `Alarm_status` tinyint(1) NOT NULL DEFAULT 0 COMMENT '报警状态  0 不报警  1 低报  2 高报',
  PRIMARY KEY (`Id`) USING BTREE,
  INDEX `key_code1`(`Result_time`, `Result_value`, `Tag_code`, `Alarm_status`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 3 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '测点采集表' ROW_FORMAT = Compact PARTITION BY HASH (`Id`)
PARTITIONS 32
(PARTITION `p0` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p1` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p10` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p11` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p12` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p13` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p14` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p15` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p16` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p17` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p18` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p19` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p2` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p20` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p21` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p22` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p23` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p24` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p25` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p26` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p27` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p28` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p29` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p3` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p30` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p31` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p4` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p5` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p6` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p7` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p8` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p9` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 )
;

-- ----------------------------
-- Records of t_rt_data
-- ----------------------------
LOAD DATA INFILE 'D:\\tmp\\query\\code\\a_dataset\\database\\t_rt_data.csv' 
INTO TABLE t_rt_data
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES;

select * from t_rt_data order by Result_time desc limit 10;


-- ----------------------------
-- Table structure for t_rt_info
-- ----------------------------
DROP TABLE IF EXISTS `t_rt_info`;
CREATE TABLE `t_rt_info`  (
  `Id` int(11) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `Tag_code` varchar(40) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '点位编码',
  `Tag_desc` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '点位名称',
  `Tag_unit` varchar(20) CHARACTER SET latin1 COLLATE latin1_swedish_ci NULL DEFAULT NULL COMMENT '单位',
  `Key_tag` tinyint(1) NOT NULL DEFAULT 0 COMMENT '关键变量  0 非关键变量  1 关键变量',
  `Unit_code` varchar(40) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '装置编码',
  `LLower_limit` double(10, 3) NULL DEFAULT NULL COMMENT '低低限报警值',
  `Lower_limit` double(10, 3) NULL DEFAULT NULL COMMENT '低限报警值',
  `Upper_limit` double(10, 3) NULL DEFAULT NULL COMMENT '高限报警值',
  `UUpper_limit` double(10, 3) NULL DEFAULT NULL COMMENT '高高限报警值',
  PRIMARY KEY (`Id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 373 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '测点定义表' ROW_FORMAT = Compact;

-- ----------------------------
-- Records of t_rt_info
-- ----------------------------
LOAD DATA INFILE 'D:\\tmp\\query\\code\\a_dataset\\database\\t_rt_info.csv' 
INTO TABLE t_rt_info
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(Id, Tag_code, @Tag_desc, Tag_unit, Key_tag, @Unit_code, @LLower_limit, @Lower_limit, @Upper_limit, @UUpper_limit)
SET
Tag_desc = IF(@Tag_desc='', NULL, @Tag_desc),
Unit_code = IF(@Unit_code='', NULL, @Unit_code),
LLower_limit = IF(@LLower_limit='', NULL, @LLower_limit),
Lower_limit = IF(@Lower_limit='', NULL, @Lower_limit),
Upper_limit = IF(@Upper_limit='', NULL, @Upper_limit),
UUpper_limit = IF(@UUpper_limit='', NULL, @UUpper_limit)
;

use db;
set session sql_mode=(SELECT REPLACE(@@sql_mode, 'ONLY_FULL_GROUP_BY', ''));
set global sql_mode=(SELECT REPLACE(@@sql_mode, 'ONLY_FULL_GROUP_BY', ''));
set session max_execution_time = 6000000;

-- 2040 s, original
select t_rt_data.Tag_code from t_rt_data where t_rt_data.Result_time > addtime( date('2023-08-31 23:59:00') , ' 5:0 ' ) group by t_rt_data.Tag_code order by max( t_rt_data.Result_value ) - min( t_rt_data.Result_value ) asc limit 1 ;
-- 1913 s (117 s)
select t_rt_data.Tag_code from t_rt_data group by t_rt_data.Tag_code order by max( t_rt_data.Result_value ) - min( t_rt_data.Result_value ) asc limit 1 ;
-- 0.047 s
select t_rt_data.Tag_code from t_rt_data where t_rt_data.Result_time > addtime( date('2023-08-31 23:59:00') , ' 5:0 ' );
-- 27.39 s
select t_rt_data.Tag_code from t_rt_data where t_rt_data.Result_time > addtime( date('2023-08-31 23:59:00') , ' 5:0 ' ) group by t_rt_data.Tag_code;
-- 2.797 s
select t_rt_data.Tag_code from t_rt_data group by t_rt_data.Tag_code;
-- 2.953 s
select t_rt_data.Tag_code from t_rt_data group by t_rt_data.Tag_code order by t_rt_data.Tag_code;


select count(*) from t_rt_info; -- 399
select count(*) from t_rt_data; -- 17140992
select count(*) from t_model_data; -- 1632
select count(*) from t_inference_data; -- 226696

--
select distinct Tag_code from t_rt_data as t1 order by (select max(t2.Result_value) - min(t2.Result_value) from t_rt_data as t2 where t2.Tag_code = t1.Tag_code) asc limit 1;
-- 31.437 s, 0202B_FR393.PV
select Tag_code from t_rt_data group by Tag_code order by max(Result_value) - min(Result_value) asc limit 1;



DROP TABLE IF EXISTS `t_rt_data`;
CREATE TABLE `t_rt_data`  (
  `Id` int(11) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `Tag_code` varchar(40) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '点位编码',
  `Result_value` double(10, 3) NULL DEFAULT NULL COMMENT '采集值',
  `Result_time` datetime(0) NULL DEFAULT NULL COMMENT '采集时间',
  `Alarm_status` tinyint(1) NOT NULL DEFAULT 0 COMMENT '报警状态  0 不报警  1 低报  2 高报',
  PRIMARY KEY (`Id`) USING BTREE,
  INDEX `key_code1`(`Result_value`, `Tag_code`, `Alarm_status`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 3 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '测点采集表' ROW_FORMAT = Compact PARTITION BY HASH (`Id`)
PARTITIONS 32
(PARTITION `p0` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p1` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p10` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p11` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p12` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p13` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p14` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p15` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p16` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p17` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p18` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p19` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p2` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p20` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p21` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p22` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p23` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p24` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p25` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p26` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p27` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p28` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p29` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p3` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p30` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p31` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p4` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p5` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p6` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p7` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p8` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 ,
PARTITION `p9` ENGINE = InnoDB MAX_ROWS = 0 MIN_ROWS = 0 )
;

-- ----------------------------
-- Records of t_rt_data
-- ----------------------------
LOAD DATA INFILE 'D:\\tmp\\query\\code\\a_dataset\\database\\t_rt_data.csv' 
INTO TABLE t_rt_data
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES;

select * from t_rt_data order by Result_time desc limit 10;

-- 31.359 s
select distinct Tag_code from t_rt_data as t1 where not exists (select * from t_rt_data as t2 where t1.Tag_code = t2.Tag_code and t2.Result_time > '2023-08-31' and t2.Alarm_status <> 0);
-- 1.500 s
select Tag_code from t_rt_data where Result_time > '2023-08-31' group by Tag_code having sum(Alarm_status <> 0) = 0;

