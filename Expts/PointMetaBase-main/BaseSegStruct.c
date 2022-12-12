BaseSeg{
  {encoder}: PointMetaBaseEncoder{
    {encoder}: Sequential{
      {0}: Sequential{
        {0}: SetAbstraction{
          {convs1}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{4, 32, kernel_size={1,}, stride={1,}}
            }
          }
        }
      }
      {1}: Sequential{
        {0}: SetAbstraction{
          {convs1}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{32, 64, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
          }
          {convs2}: Sequential{
            {0}: Sequential{
              {0}: Conv2d{3, 64, kernel_size={1, 1}, stride={1, 1}, bias=False}
              {1}: BatchNorm2d{64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
          }
          {grouper}: QueryAndGroup{}
        }
        {1}: InvResMLP{
          {convs}: LocalAggregation{
            {convs1}: Sequential{
              {0}: Sequential{
                {0}: Conv1d{64, 64, kernel_size={1,}, stride={1,}, bias=False}
                {1}: BatchNorm1d{64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
                {2}: ReLU{inplace=True}
              }
            }
            {grouper}: QueryAndGroup{}
          }
          {pwconv}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{64, 64, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
            {1}: Sequential{
              {0}: Conv1d{64, 64, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
            }
          }
          {act}: ReLU{inplace=True}
        }
        {2}: InvResMLP{
          {convs}: LocalAggregation{
            {convs1}: Sequential{
              {0}: Sequential{
                {0}: Conv1d{64, 64, kernel_size={1,}, stride={1,}, bias=False}
                {1}: BatchNorm1d{64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
                {2}: ReLU{inplace=True}
              }
            }
            {grouper}: QueryAndGroup{}
          }
          {pwconv}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{64, 64, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
            {1}: Sequential{
              {0}: Conv1d{64, 64, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
            }
          }
          {act}: ReLU{inplace=True}
        }
      }
      {2}: Sequential{
        {0}: SetAbstraction{
          {convs1}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{64, 128, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
          }
          {convs2}: Sequential{
            {0}: Sequential{
              {0}: Conv2d{3, 128, kernel_size={1, 1}, stride={1, 1}, bias=False}
              {1}: BatchNorm2d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
          }
          {grouper}: QueryAndGroup{}
        }
        {1}: InvResMLP{
          {convs}: LocalAggregation{
            {convs1}: Sequential{
              {0}: Sequential{
                {0}: Conv1d{128, 128, kernel_size={1,}, stride={1,}, bias=False}
                {1}: BatchNorm1d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
                {2}: ReLU{inplace=True}
              }
            }
            {grouper}: QueryAndGroup{}
          }
          {pwconv}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{128, 128, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
            {1}: Sequential{
              {0}: Conv1d{128, 128, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
            }
          }
          {act}: ReLU{inplace=True}
        }
        {2}: InvResMLP{
          {convs}: LocalAggregation{
            {convs1}: Sequential{
              {0}: Sequential{
                {0}: Conv1d{128, 128, kernel_size={1,}, stride={1,}, bias=False}
                {1}: BatchNorm1d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
                {2}: ReLU{inplace=True}
              }
            }
            {grouper}: QueryAndGroup{}
          }
          {pwconv}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{128, 128, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
            {1}: Sequential{
              {0}: Conv1d{128, 128, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
            }
          }
          {act}: ReLU{inplace=True}
        }
        {3}: InvResMLP{
          {convs}: LocalAggregation{
            {convs1}: Sequential{
              {0}: Sequential{
                {0}: Conv1d{128, 128, kernel_size={1,}, stride={1,}, bias=False}
                {1}: BatchNorm1d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
                {2}: ReLU{inplace=True}
              }
            }
            {grouper}: QueryAndGroup{}
          }
          {pwconv}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{128, 128, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
            {1}: Sequential{
              {0}: Conv1d{128, 128, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
            }
          }
          {act}: ReLU{inplace=True}
        }
        {4}: InvResMLP{
          {convs}: LocalAggregation{
            {convs1}: Sequential{
              {0}: Sequential{
                {0}: Conv1d{128, 128, kernel_size={1,}, stride={1,}, bias=False}
                {1}: BatchNorm1d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
                {2}: ReLU{inplace=True}
              }
            }
            {grouper}: QueryAndGroup{}
          }
          {pwconv}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{128, 128, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
            {1}: Sequential{
              {0}: Conv1d{128, 128, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
            }
          }
          {act}: ReLU{inplace=True}
        }
      }
      {3}: Sequential{
        {0}: SetAbstraction{
          {convs1}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{128, 256, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
          }
          {convs2}: Sequential{
            {0}: Sequential{
              {0}: Conv2d{3, 256, kernel_size={1, 1}, stride={1, 1}, bias=False}
              {1}: BatchNorm2d{256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
          }
          {grouper}: QueryAndGroup{}
        }
        {1}: InvResMLP{
          {convs}: LocalAggregation{
            {convs1}: Sequential{
              {0}: Sequential{
                {0}: Conv1d{256, 256, kernel_size={1,}, stride={1,}, bias=False}
                {1}: BatchNorm1d{256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
                {2}: ReLU{inplace=True}
              }
            }
            {grouper}: QueryAndGroup{}
          }
          {pwconv}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{256, 256, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
            {1}: Sequential{
              {0}: Conv1d{256, 256, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
            }
          }
          {act}: ReLU{inplace=True}
        }
        {2}: InvResMLP{
          {convs}: LocalAggregation{
            {convs1}: Sequential{
              {0}: Sequential{
                {0}: Conv1d{256, 256, kernel_size={1,}, stride={1,}, bias=False}
                {1}: BatchNorm1d{256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
                {2}: ReLU{inplace=True}
              }
            }
            {grouper}: QueryAndGroup{}
          }
          {pwconv}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{256, 256, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
            {1}: Sequential{
              {0}: Conv1d{256, 256, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
            }
          }
          {act}: ReLU{inplace=True}
        }
      }
      {4}: Sequential{
        {0}: SetAbstraction{
          {convs1}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{256, 512, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
          }
          {convs2}: Sequential{
            {0}: Sequential{
              {0}: Conv2d{3, 512, kernel_size={1, 1}, stride={1, 1}, bias=False}
              {1}: BatchNorm2d{512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
          }
          {grouper}: QueryAndGroup{}
        }
        {1}: InvResMLP{
          {convs}: LocalAggregation{
            {convs1}: Sequential{
              {0}: Sequential{
                {0}: Conv1d{512, 512, kernel_size={1,}, stride={1,}, bias=False}
                {1}: BatchNorm1d{512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
                {2}: ReLU{inplace=True}
              }
            }
            {grouper}: QueryAndGroup{}
          }
          {pwconv}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{512, 512, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
            {1}: Sequential{
              {0}: Conv1d{512, 512, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
            }
          }
          {act}: ReLU{inplace=True}
        }
        {2}: InvResMLP{
          {convs}: LocalAggregation{
            {convs1}: Sequential{
              {0}: Sequential{
                {0}: Conv1d{512, 512, kernel_size={1,}, stride={1,}, bias=False}
                {1}: BatchNorm1d{512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
                {2}: ReLU{inplace=True}
              }
            }
            {grouper}: QueryAndGroup{}
          }
          {pwconv}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{512, 512, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
            {1}: Sequential{
              {0}: Conv1d{512, 512, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
            }
          }
          {act}: ReLU{inplace=True}
        }
      }
    }
    {pe_encoder}: ModuleList{
      {0}: ModuleList{}
      {1}: Sequential{
        {0}: Sequential{
          {0}: Conv2d{3, 64, kernel_size={1, 1}, stride={1, 1}, bias=False}
          {1}: BatchNorm2d{64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
          {2}: ReLU{inplace=True}
        }
      }
      {2}: Sequential{
        {0}: Sequential{
          {0}: Conv2d{3, 128, kernel_size={1, 1}, stride={1, 1}, bias=False}
          {1}: BatchNorm2d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
          {2}: ReLU{inplace=True}
        }
      }
      {3}: Sequential{
        {0}: Sequential{
          {0}: Conv2d{3, 256, kernel_size={1, 1}, stride={1, 1}, bias=False}
          {1}: BatchNorm2d{256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
          {2}: ReLU{inplace=True}
        }
      }
      {4}: Sequential{
        {0}: Sequential{
          {0}: Conv2d{3, 512, kernel_size={1, 1}, stride={1, 1}, bias=False}
          {1}: BatchNorm2d{512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
          {2}: ReLU{inplace=True}
        }
      }
    }
  }
  {decoder}: PointNextDecoder{
    {decoder}: Sequential{
      {0}: Sequential{
        {0}: FeaturePropogation{
          {convs}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{96, 32, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
            {1}: Sequential{
              {0}: Conv1d{32, 32, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
          }
        }
      }
      {1}: Sequential{
        {0}: FeaturePropogation{
          {convs}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{192, 64, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
            {1}: Sequential{
              {0}: Conv1d{64, 64, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
          }
        }
      }
      {2}: Sequential{
        {0}: FeaturePropogation{
          {convs}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{384, 128, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
            {1}: Sequential{
              {0}: Conv1d{128, 128, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
          }
        }
      }
      {3}: Sequential{
        {0}: FeaturePropogation{
          {convs}: Sequential{
            {0}: Sequential{
              {0}: Conv1d{768, 256, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
            {1}: Sequential{
              {0}: Conv1d{256, 256, kernel_size={1,}, stride={1,}, bias=False}
              {1}: BatchNorm1d{256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
              {2}: ReLU{inplace=True}
            }
          }
        }
      }
    }
  }
  {head}: SegHead{
    {head}: Sequential{
      {0}: Sequential{
        {0}: Conv1d{32, 32, kernel_size={1,}, stride={1,}, bias=False}
        {1}: BatchNorm1d{32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True}
        {2}: ReLU{inplace=True}
      }
      {1}: Dropout{p=0.5, inplace=False}
      {2}: Sequential{
        {0}: Conv1d{32, 13, kernel_size={1,}, stride={1,}}
      }
    }
  }
}


