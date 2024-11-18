from yacs.config import CfgNode as CN

tank_cfg = CN()

tank_cfg.META_ARC = "tank_test_config"

tank_cfg.scenes = (
    "Family", "Francis", "Horse", "Lighthouse", "M60", "Panther", "Playground", "Train",    # intermediate
    "Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple"                     # advanced
)

tank_cfg.Family = CN()
tank_cfg.Family.max_h = 1080
tank_cfg.Family.max_w = 2048
tank_cfg.Family.conf = [0.4, 0.6, 0.9]
tank_cfg.Family.geo_mask_thres = 6
tank_cfg.Family.photo_thres = 0.9
tank_cfg.Family.geo_pixel_thres = 0.75
tank_cfg.Family.geo_depth_thres = 0.01
tank_cfg.Family.endrange_filter = False

tank_cfg.Francis = CN()
tank_cfg.Francis.max_h = 1080
tank_cfg.Francis.max_w = 2048
tank_cfg.Francis.conf = [0.4, 0.6, 0.95]
tank_cfg.Francis.geo_mask_thres = 8
tank_cfg.Francis.photo_thres = 0.8
tank_cfg.Francis.geo_pixel_thres = 1.0
tank_cfg.Francis.geo_depth_thres = 0.01
tank_cfg.Francis.endrange_filter = False

tank_cfg.Horse = CN()
tank_cfg.Horse.max_h = 1080
tank_cfg.Horse.max_w = 2048
tank_cfg.Horse.conf = [0.05, 0.1, 0.6]
tank_cfg.Horse.geo_mask_thres = 4
tank_cfg.Horse.photo_thres = 0.8
tank_cfg.Horse.geo_pixel_thres = 1.25
tank_cfg.Horse.geo_depth_thres = 0.01
tank_cfg.Horse.endrange_filter = False

tank_cfg.Lighthouse = CN()
tank_cfg.Lighthouse.max_h = 1080
tank_cfg.Lighthouse.max_w = 2048
tank_cfg.Lighthouse.conf = [0.5, 0.6, 0.9]
tank_cfg.Lighthouse.geo_mask_thres = 7
tank_cfg.Lighthouse.photo_thres = 0.8
tank_cfg.Lighthouse.geo_pixel_thres = 1.0
tank_cfg.Lighthouse.geo_depth_thres = 0.01
tank_cfg.Lighthouse.endrange_filter = False

tank_cfg.M60 = CN()
tank_cfg.M60.max_h = 1080
tank_cfg.M60.max_w = 2048
tank_cfg.M60.conf = [0.4, 0.7, 0.9]
tank_cfg.M60.geo_mask_thres = 6
tank_cfg.M60.photo_thres = 0.9
tank_cfg.M60.geo_pixel_thres = 0.75
tank_cfg.M60.geo_depth_thres = 0.005
tank_cfg.M60.endrange_filter = False

tank_cfg.Panther = CN()
tank_cfg.Panther.max_h = 896
tank_cfg.Panther.max_w = 1216
tank_cfg.Panther.conf = [0.1, 0.15, 0.9]
tank_cfg.Panther.geo_mask_thres = 6
tank_cfg.Panther.photo_thres = 0.9
tank_cfg.Panther.geo_pixel_thres = 1.0
tank_cfg.Panther.geo_depth_thres = 0.01
tank_cfg.Panther.endrange_filter = False

tank_cfg.Playground = CN()
tank_cfg.Playground.max_h = 1080
tank_cfg.Playground.max_w = 2048
tank_cfg.Playground.conf = [0.5, 0.7, 0.9]
tank_cfg.Playground.geo_mask_thres = 7
tank_cfg.Playground.photo_thres = 0.85
tank_cfg.Playground.geo_pixel_thres = 1.0
tank_cfg.Playground.geo_depth_thres = 0.01
tank_cfg.Playground.endrange_filter = False

tank_cfg.Train = CN()
tank_cfg.Train.max_h = 1080
tank_cfg.Train.max_w = 2048
tank_cfg.Train.conf = [0.3, 0.6, 0.95]
tank_cfg.Train.geo_mask_thres = 6
tank_cfg.Train.photo_thres = 0.9
tank_cfg.Train.geo_pixel_thres = 1.5
tank_cfg.Train.geo_depth_thres = 0.01
tank_cfg.Train.endrange_filter = False

tank_cfg.Auditorium = CN()
tank_cfg.Auditorium.max_h = 1080
tank_cfg.Auditorium.max_w = 2048
tank_cfg.Auditorium.conf = [0.0, 0.0, 0.4]
tank_cfg.Auditorium.geo_mask_thres = 3
tank_cfg.Auditorium.photo_thres = 0.7
tank_cfg.Auditorium.geo_pixel_thres = 4.0
tank_cfg.Auditorium.geo_depth_thres = 0.005
tank_cfg.Auditorium.endrange_filter = False

tank_cfg.Ballroom = CN()
tank_cfg.Ballroom.max_h = 1080
tank_cfg.Ballroom.max_w = 2048
tank_cfg.Ballroom.conf = [0.0, 0.0, 0.5]
tank_cfg.Ballroom.geo_mask_thres = 4
tank_cfg.Ballroom.photo_thres = 0.8
tank_cfg.Ballroom.geo_pixel_thres = 4.0
tank_cfg.Ballroom.geo_depth_thres = 0.005
tank_cfg.Ballroom.endrange_filter = False

tank_cfg.Courtroom = CN()
tank_cfg.Courtroom.max_h = 1080
tank_cfg.Courtroom.max_w = 2048
tank_cfg.Courtroom.conf = [0.0, 0.0, 0.4]
tank_cfg.Courtroom.geo_mask_thres = 3
tank_cfg.Courtroom.photo_thres = 0.8
tank_cfg.Courtroom.geo_pixel_thres = 3.0
tank_cfg.Courtroom.geo_depth_thres = 0.005
tank_cfg.Courtroom.endrange_filter = False

tank_cfg.Museum = CN()
tank_cfg.Museum.max_h = 1080
tank_cfg.Museum.max_w = 2048
tank_cfg.Museum.conf = [0.0, 0.0, 0.7]
tank_cfg.Museum.geo_mask_thres = 4
tank_cfg.Museum.photo_thres = 0.8
tank_cfg.Museum.geo_pixel_thres = 4.0
tank_cfg.Museum.geo_depth_thres = 0.01
tank_cfg.Museum.endrange_filter = False

tank_cfg.Palace = CN()
tank_cfg.Palace.max_h = 1080
tank_cfg.Palace.max_w = 2048
tank_cfg.Palace.conf = [0.0, 0.0, 0.7]
tank_cfg.Palace.geo_mask_thres = 5
tank_cfg.Palace.photo_thres = 0.9
tank_cfg.Palace.geo_pixel_thres = 4.0
tank_cfg.Palace.geo_depth_thres = 0.005
tank_cfg.Palace.endrange_filter = True

tank_cfg.Temple = CN()
tank_cfg.Temple.max_h = 1080
tank_cfg.Temple.max_w = 2048
tank_cfg.Temple.conf = [0.0, 0.0, 0.4]
tank_cfg.Temple.geo_mask_thres = 3
tank_cfg.Temple.photo_thres = 0.8
tank_cfg.Temple.geo_pixel_thres = 4.0
tank_cfg.Temple.geo_depth_thres = 0.01
tank_cfg.Temple.endrange_filter = True
