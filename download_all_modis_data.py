
import ee
import time
import sys

ee.Initialize()



def export_oneimage(img,location,name,scale,crs):
    task = ee.batch.Export.image(img, name, {
      'driveFolder':location,
      'driveFileNamePrefix':name,
      'outputBucket': location,
      'outputPrefix': name,
      'scale':scale,
      'crs':crs
    })
    task.start()
    while task.status()['state'] == 'RUNNING':
    print 'Running...'
    # Perhaps task.cancel() at some point.
    time.sleep(10)
    print 'Done.', task.status()




def appendBand(current, previous):
    # Rename the band
    previous=ee.Image(previous)
    current = current.select([0,1,2,3,4,5,6])
    # Append it to the result (Note: only return current item on first element/iteration)
    accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous,None), current, previous.addBands(ee.Image(current)))
    # Return the accumulation
    return accum


fusion_table_dict = {
#    'ET_admin2': ("users/tkenned97/ET_Admin2_simp_2k",'2001-1-1','2016','ADMIN2'),
#     'TZ_admin1': ("users/tkenned97/TZ_Admin1_simp_2k", '2000-2-18','2015','ADMIN1'),
#     'ZM_admin2': ("users/tkenned97/ZM_Admin2_simp_2k",'2000-2-18','2017','ADMIN2'),
#     'NG_admin1': ("users/tkenned97/NG_Admin1_simp_2k",'2000-2-18','2017','ADMIN1'),
#     'MW_admin2': ("users/tkenned97/MW_Admin2_simp_2k",'2000-2-18','2017','ADMIN2'),
#     'KE_admin2': ("users/tkenned97/KE_Admin2_simp_2k",'2000-2-18','2016','ADMIN2')
      'SU_admin2': ('users/akkstuff/sudan_admin2', '2013-1-1', '2018', 'ADM2_EN')
}





for country_key,val in fusion_table_dict.items():
    fid, start_date, end_date, admin_type = val
    region = ee.FeatureCollection(fid)
    imgcoll = ee.ImageCollection('MODIS/006/MOD09A1').filterDate(start_date, end_date + '-12-31').sort('system:time_start').filterBounds(ee.Geometry.Rectangle(-22, 38, 60, -38))

    img=imgcoll.iterate(appendBand)
    img=ee.Image(img)

    img_0=ee.Image(ee.Number(0))
    img_5000=ee.Image(ee.Number(5000))

    img=img.min(img_5000)
    img=img.max(img_0)

    scale  = 500
    crs='EPSG:4326'

    raw_feature_data = list(region.toList(9999999).getInfo())
    feature_names = []
    for x in raw_feature_data:
        feature_names.append(x['properties'][admin_type])


    for name in feature_names:
        feature = region.filterMetadata(admin_type, 'equals', name)

        if "'" in name:
            name = name.replace("'","")
        if "/" in name:
            name = name.replace("/","")

        feature = ee.Feature(feature.first())
        export_oneimage(img.clip(feature), 'africa-yield', country_key + "_" + name, scale, crs)







