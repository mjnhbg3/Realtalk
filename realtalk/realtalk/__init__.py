from .realtalk import RealTalk


async def setup(bot):
    """Load the RealTalk cog."""
    cog = RealTalk(bot)
    await bot.add_cog(cog)