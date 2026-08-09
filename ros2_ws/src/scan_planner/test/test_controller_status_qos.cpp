#include <gtest/gtest.h>

#include <rmw/types.h>

#include "plan_manage/controller_status_qos.h"

namespace scan_planner
{
namespace
{

TEST(ControllerStatusQos, MatchesBoundedLateJoinEvidenceContract)
{
  const auto qos = makeControllerStatusEvidenceQos();
  const auto profile = qos.get_rmw_qos_profile();

  EXPECT_EQ(profile.history, RMW_QOS_POLICY_HISTORY_KEEP_LAST);
  EXPECT_EQ(
      profile.depth,
      static_cast<std::size_t>(kControllerStatusEvidenceDepth));
  EXPECT_EQ(profile.reliability, RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  EXPECT_EQ(profile.durability, RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
}

TEST(ControllerStatusQos, RejectsDepthDifferentFromEvidenceContract)
{
  EXPECT_NO_THROW(
      validateControllerStatusEvidenceDepth(kControllerStatusEvidenceDepth));
  EXPECT_THROW(validateControllerStatusEvidenceDepth(1), std::runtime_error);
  EXPECT_THROW(validateControllerStatusEvidenceDepth(63), std::runtime_error);
  EXPECT_THROW(validateControllerStatusEvidenceDepth(65), std::runtime_error);
}

}  // namespace
}  // namespace scan_planner
