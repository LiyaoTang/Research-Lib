Header header
uint8 mark_type      #/*!< 类型 LaneMarkingType*/
uint8 mark_color     #/*!< 类型 LaneMarkingColor */
float64 a
float64 b
float64 c
float64 d            # /* y = a * pow(x,3) + b * pow(x,2) + c * x + d */
float64 x_start      # /*!< 纵向起始点(米) */
float64 x_end        # /*!< 纵向终止点(米) */

float64 image_a
float64 image_b
float64 image_c
float64 image_d
float64 image_x_start
float64 image_x_end
