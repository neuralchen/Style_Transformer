import  argparse

def str2bool(v):
    return v.lower() in ('true')

def getParameters():

    parser = argparse.ArgumentParser()

    # Basic information
    parser.add_argument('--content', type=str, default='decoder', choices=['decoder', 'test'])
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--version', type=str, default='test4')
    parser.add_argument('--description', type=str, default='修改style的系数')
    # AE training setting
    parser.add_argument('--total_epoch', type=int, default=50)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--channels', type=int, default=24)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--content_dir', type=str, default='D:\\Downloads\\data_large')
    parser.add_argument('--style_dir', type=str, default='D:\\F_Disk\\data_set\\Art_Data\\data_art_backup')
    parser.add_argument('--fix_data_dir', type=str, default='./fix_content')
    
    parser.add_argument('--script_name', type=str, default='deepconv')
    parser.add_argument('--save_dir', type=str, default='./logs')
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--style_weight', type=float, default=150000)
    parser.add_argument('--content_weight', type=float, default=1)
    parser.add_argument('--image_crop_size', type=int, default=512)
    parser.add_argument('--selected_content_dir', nargs='+', help='selected style dir for training', 
        default=['a/abbey', 'a/arch', 'a/amphitheater', 'a/aqueduct', 'a/arena/rodeo', 'a/athletic_field/outdoor',
         'b/badlands', 'b/balcony/exterior', 'b/bamboo_forest', 'b/barn', 'b/barndoor', 'b/baseball_field',
         'b/basilica', 'b/bayou', 'b/beach', 'b/beach_house', 'b/beer_garden', 'b/boardwalk', 'b/boathouse',
         'b/botanical_garden', 'b/bullring', 'b/butte', 'c/cabin/outdoor', 'c/campsite', 'c/campus',
         'c/canal/natural', 'c/canal/urban', 'c/canyon', 'c/castle', 'c/church/outdoor', 'c/chalet',
         'c/cliff', 'c/coast', 'c/corn_field', 'c/corral', 'c/cottage', 'c/courtyard', 'c/crevasse',
         'd/dam', 'd/desert/vegetation', 'd/desert_road', 'd/doorway/outdoor', 'f/farm', 'f/fairway',
         'f/field/cultivated', 'f/field/wild', 'f/field_road', 'f/fishpond', 'f/florist_shop/indoor',
         'f/forest/broadleaf', 'f/forest_path', 'f/forest_road', 'f/formal_garden', 'g/gazebo/exterior',
         'g/glacier', 'g/golf_course', 'g/greenhouse/indoor', 'g/greenhouse/outdoor', 'g/grotto', 'g/gorge',
         'h/hayfield', 'h/herb_garden', 'h/hot_spring', 'h/house', 'h/hunting_lodge/outdoor', 'i/ice_floe',
         'i/ice_shelf', 'i/iceberg', 'i/inn/outdoor', 'i/islet', 'j/japanese_garden', 'k/kasbah',
         'k/kennel/outdoor', 'l/lagoon', 'l/lake/natural', 'l/lawn', 'l/library/outdoor', 'l/lighthouse',
         'm/mansion', 'm/marsh', 'm/mausoleum', 'm/moat/water', 'm/mosque/outdoor', 'm/mountain',
         'm/mountain_path', 'm/mountain_snowy', 'o/oast_house', 'o/ocean', 'o/orchard', 'p/park',
         'p/pasture', 'p/pavilion', 'p/picnic_area', 'p/pier', 'p/pond', 'r/raft', 'r/railroad_track',
         'r/rainforest', 'r/rice_paddy', 'r/river', 'r/rock_arch', 'r/roof_garden', 'r/rope_bridge',
         'r/ruin', 's/schoolhouse', 's/sky', 's/snowfield', 's/swamp', 's/swimming_hole',
         's/synagogue/outdoor', 't/temple/asia', 't/topiary_garden', 't/tree_farm', 't/tree_house',
         'u/underwater/ocean_deep', 'u/utility_room', 'v/valley', 'v/vegetable_garden', 'v/viaduct',
         'v/village', 'v/vineyard', 'v/volcano', 'w/waterfall', 'w/watering_hole', 'w/wave',
         'w/wheat_field', 'z/zen_garden', 'a/alcove', 'a/apartment-building/outdoor', 'a/artists_loft',
         'b/building_facade', 'c/cemetery'])

    parser.add_argument('--selected_style_dir', nargs='+', help='selected content dir for training',
        default=["vangogh", "samuel", "picasso", "kandinsky", "monet", "nicholas", "berthe-morisot"])
    
    # Test setting
    parser.add_argument('--test_epoch', type=int, default=100)
    parser.add_argument('--test_batch', type=int, default=9900)

    return parser.parse_args()

