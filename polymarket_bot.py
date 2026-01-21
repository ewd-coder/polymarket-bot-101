    async def _analyze_market(self, market: Dict) -> List[MarketOpportunity]:
        """Analyze a single market for opportunities"""
        market_id = market.get('condition_id')
        if not market_id:
            return []
        
        opportunities = []
        
        # Get market data
        orderbook = await self.api.get_market_orderbook(market)
        market_stats = await self.api.get_market_stats(market_id)
        
        # Skip if no valid orderbook
        if not orderbook.get('bids') or not orderbook.get('asks'):
            return []
        
        # Calculate key metrics
        metrics = self._calculate_metrics(market, orderbook, market_stats)
        
        # Check for different opportunity types
        
        # 1. High Spread Opportunity
        if metrics['spread'] >= Decimal('0.02'):  # 2%+ spread
            score = self._score_high_spread(metrics)
            if score > 0.6:
                opportunities.append(MarketOpportunity(
                    market_id=market_id,
                    market_name=market.get('question', 'Unknown'),
                    opportunity_type=OpportunityType.HIGH_SPREAD,
                    score=score,
                    liquidity=metrics['liquidity'],
                    current_spread=metrics['spread'],
                    volatility=metrics['volatility'],
                    volume_24h=metrics['volume_24h'],
                    time_to_resolution=metrics['days_to_resolution'],
                    metadata={'mid_price': metrics['mid_price']}
                ))
        
        # 2. High Liquidity + Low Volatility (ideal for market making)
        if metrics['liquidity'] >= Decimal('200000') and metrics['volatility'] < Decimal('0.05'):
            score = self._score_stable_liquid(metrics)
            if score > 0.7:
                opportunities.append(MarketOpportunity(
                    market_id=market_id,
                    market_name=market.get('question', 'Unknown'),
                    opportunity_type=OpportunityType.HIGH_LIQUIDITY,
                    score=score,
                    liquidity=metrics['liquidity'],
                    current_spread=metrics['spread'],
                    volatility=metrics['volatility'],
                    volume_24h=metrics['volume_24h'],
                    time_to_resolution=metrics['days_to_resolution'],
                    metadata={'depth': metrics['order_depth']}
                ))
        
        # 3. Mean Reversion Opportunity
        if self._detect_price_spike(market_id, metrics['mid_price']):
            score = self._score_mean_reversion(metrics)
            if score > 0.65:
                opportunities.append(MarketOpportunity(
                    market_id=market_id,
                    market_name=market.get('question', 'Unknown'),
                    opportunity_type=OpportunityType.MEAN_REVERSION,
                    score=score,
                    liquidity=metrics['liquidity'],
                    current_spread=metrics['spread'],
                    volatility=metrics['volatility'],
                    volume_24h=metrics['volume_24h'],
                    time_to_resolution=metrics['days_to_resolution'],
                    metadata={
                        'current_price': metrics['mid_price'],
                        'avg_price': metrics['avg_price_24h']
                    }
                ))
        
        # 4. Low Volatility Stable Markets
        if metrics['volatility'] < Decimal('0.02') and metrics['volume_24h'] > Decimal('10import asyncio
import logging
from decimal import Decimal
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
from web3 import Web3
from collections import defaultdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpportunityType(Enum):
    """Types of trading opportunities"""
    HIGH_SPREAD = "high_spread"
    HIGH_LIQUIDITY = "high_liquidity"
    MEAN_REVERSION = "mean_reversion"
    LOW_VOLATILITY = "low_volatility"
    ARBITRAGE = "arbitrage"


@dataclass
class MarketOpportunity:
    """Represents a trading opportunity"""
    market_id: str
    market_name: str
    opportunity_type: OpportunityType
    score: float
    liquidity: Decimal
    current_spread: Decimal
    volatility: Decimal
    volume_24h: Decimal
    time_to_resolution: Optional[int]  # days
    metadata: Dict
    
    def __lt__(self, other):
        return self.score < other.score


@dataclass
class MarketConfig:
    """Configuration for individual market"""
    condition_id: str
    min_liquidity: Decimal
    max_position_size: Decimal
    spread_bps: int  # basis points
    rebalance_threshold: Decimal


@dataclass
class Position:
    """Track position in a market"""
    market_id: str
    yes_shares: Decimal
    no_shares: Decimal
    avg_yes_price: Decimal
    avg_no_price: Decimal
    realized_pnl: Decimal
    
    @property
    def inventory_skew(self) -> Decimal:
        """Calculate inventory imbalance (-1 to 1)"""
        total = self.yes_shares + self.no_shares
        if total == 0:
            return Decimal(0)
        return (self.yes_shares - self.no_shares) / total
    
    @property
    def net_exposure(self) -> Decimal:
        """Calculate net directional exposure"""
        return self.yes_shares - self.no_shares


class RiskManager:
    """Manages risk limits and position sizing"""
    
    def __init__(self, total_capital: Decimal, max_position_pct: Decimal = Decimal('0.03')):
        self.total_capital = total_capital
        self.max_position_pct = max_position_pct
        self.max_position_size = total_capital * max_position_pct
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = Decimal(0)
        self.max_daily_loss = total_capital * Decimal('0.05')  # 5% daily loss limit
        
    def can_open_position(self, market_id: str, size: Decimal) -> bool:
        """Check if new position is within risk limits"""
        if self.daily_pnl < -self.max_daily_loss:
            logger.warning("Daily loss limit reached")
            return False
            
        current_position = self.positions.get(market_id)
        if current_position:
            total_exposure = abs(current_position.net_exposure) + size
            if total_exposure > self.max_position_size:
                return False
        else:
            if size > self.max_position_size:
                return False
                
        return True
    
    def update_position(self, market_id: str, side: str, shares: Decimal, price: Decimal):
        """Update position after trade execution"""
        if market_id not in self.positions:
            self.positions[market_id] = Position(
                market_id=market_id,
                yes_shares=Decimal(0),
                no_shares=Decimal(0),
                avg_yes_price=Decimal(0),
                avg_no_price=Decimal(0),
                realized_pnl=Decimal(0)
            )
        
        pos = self.positions[market_id]
        
        if side == 'YES':
            total_cost = pos.yes_shares * pos.avg_yes_price + shares * price
            pos.yes_shares += shares
            pos.avg_yes_price = total_cost / pos.yes_shares if pos.yes_shares > 0 else Decimal(0)
        else:
            total_cost = pos.no_shares * pos.avg_no_price + shares * price
            pos.no_shares += shares
            pos.avg_no_price = total_cost / pos.no_shares if pos.no_shares > 0 else Decimal(0)
    
    def needs_rebalancing(self, market_id: str, threshold: Decimal = Decimal('0.3')) -> bool:
        """Check if position needs rebalancing"""
        if market_id not in self.positions:
            return False
        return abs(self.positions[market_id].inventory_skew) > threshold


class MarketScanner:
    """Scans and identifies profitable trading opportunities"""
    
    def __init__(self, api: PolymarketAPI):
        self.api = api
        self.price_history: Dict[str, List[tuple]] = defaultdict(list)  # market_id -> [(timestamp, price)]
        self.opportunity_cache: List[MarketOpportunity] = []
        self.last_scan: Optional[datetime] = None
        
    async def scan_all_markets(self) -> List[MarketOpportunity]:
        """Comprehensive scan of all available markets"""
        logger.info("Starting market scan...")
        
        # Fetch all active markets
        all_markets = await self.api.get_all_markets()
        
        opportunities = []
        
        for market in all_markets:
            try:
                # Skip closed or invalid markets
                if not self._is_valid_market(market):
                    continue
                
                # Analyze market for opportunities
                market_opps = await self._analyze_market(market)
                opportunities.extend(market_opps)
                
            except Exception as e:
                logger.error(f"Error analyzing market {market.get('id')}: {e}")
                continue
        
        # Sort by score (highest first)
        opportunities.sort(reverse=True)
        
        self.opportunity_cache = opportunities[:50]  # Keep top 50
        self.last_scan = datetime.utcnow()
        
        logger.info(f"Scan complete. Found {len(opportunities)} opportunities")
        return self.opportunity_cache
    
    def _is_valid_market(self, market: Dict) -> bool:
        """Check if market meets basic criteria"""
        return (
            not market.get('closed', True) and
            market.get('outcome_count', 0) == 2 and  # Binary only
            Decimal(market.get('liquidity', 0)) >= Decimal('50000') and
            market.get('active', True) and
            not market.get('resolved', False) and
            self._has_clear_resolution(market)
        )
    
    def _has_clear_resolution(self, market: Dict) -> bool:
        """Check if market has clear resolution criteria"""
        # Avoid subjective or unclear markets
        blacklist_keywords = ['opinion', 'should', 'best', 'most popular']
        question = market.get('question', '').lower()
        
        for keyword in blacklist_keywords:
            if keyword in question:
                return False
        
        # Prefer markets with specific dates
        return market.get('end_date') is not None
    
    async def _analyze_market(self, market: Dict) -> List[MarketOpportunity]:
        """Analyze a single market for opportunities"""
        market_id = market['id']
        opportunities = []
        
        # Get market data
        orderbook = await self.api.get_orderbook(market_id)
        market_stats = await self.api.get_market_stats(market_id)
        
        # Calculate key metrics
        metrics = self._calculate_metrics(market, orderbook, market_stats)
        
        # Check for different opportunity types
        
        # 1. High Spread Opportunity
        if metrics['spread'] >= Decimal('0.02'):  # 2%+ spread
            score = self._score_high_spread(metrics)
            if score > 0.6:
                opportunities.append(MarketOpportunity(
                    market_id=market_id,
                    market_name=market['question'],
                    opportunity_type=OpportunityType.HIGH_SPREAD,
                    score=score,
                    liquidity=metrics['liquidity'],
                    current_spread=metrics['spread'],
                    volatility=metrics['volatility'],
                    volume_24h=metrics['volume_24h'],
                    time_to_resolution=metrics['days_to_resolution'],
                    metadata={'mid_price': metrics['mid_price']}
                ))
        
        # 2. High Liquidity + Low Volatility (ideal for market making)
        if metrics['liquidity'] >= Decimal('200000') and metrics['volatility'] < Decimal('0.05'):
            score = self._score_stable_liquid(metrics)
            if score > 0.7:
                opportunities.append(MarketOpportunity(
                    market_id=market_id,
                    market_name=market['question'],
                    opportunity_type=OpportunityType.HIGH_LIQUIDITY,
                    score=score,
                    liquidity=metrics['liquidity'],
                    current_spread=metrics['spread'],
                    volatility=metrics['volatility'],
                    volume_24h=metrics['volume_24h'],
                    time_to_resolution=metrics['days_to_resolution'],
                    metadata={'depth': metrics['order_depth']}
                ))
        
        # 3. Mean Reversion Opportunity
        if self._detect_price_spike(market_id, metrics['mid_price']):
            score = self._score_mean_reversion(metrics)
            if score > 0.65:
                opportunities.append(MarketOpportunity(
                    market_id=market_id,
                    market_name=market['question'],
                    opportunity_type=OpportunityType.MEAN_REVERSION,
                    score=score,
                    liquidity=metrics['liquidity'],
                    current_spread=metrics['spread'],
                    volatility=metrics['volatility'],
                    volume_24h=metrics['volume_24h'],
                    time_to_resolution=metrics['days_to_resolution'],
                    metadata={
                        'current_price': metrics['mid_price'],
                        'avg_price': metrics['avg_price_24h']
                    }
                ))
        
        # 4. Low Volatility Stable Markets
        if metrics['volatility'] < Decimal('0.02') and metrics['volume_24h'] > Decimal('10000'):
            score = self._score_low_volatility(metrics)
            if score > 0.75:
                opportunities.append(MarketOpportunity(
                    market_id=market_id,
                    market_name=market['question'],
                    opportunity_type=OpportunityType.LOW_VOLATILITY,
                    score=score,
                    liquidity=metrics['liquidity'],
                    current_spread=metrics['spread'],
                    volatility=metrics['volatility'],
                    volume_24h=metrics['volume_24h'],
                    time_to_resolution=metrics['days_to_resolution'],
                    metadata={'price_stability': metrics['price_range_24h']}
                ))
        
        return opportunities
    
    def _calculate_metrics(self, market: Dict, orderbook: Dict, stats: Dict) -> Dict:
        """Calculate key market metrics"""
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return self._default_metrics()
        
        best_bid = Decimal(str(bids[0]['price']))
        best_ask = Decimal(str(asks[0]['price']))
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        # Calculate order depth (liquidity within 2% of mid)
        depth_range = mid_price * Decimal('0.02')
        bid_depth = sum(
            Decimal(str(b['size'])) 
            for b in bids 
            if Decimal(str(b['price'])) >= mid_price - depth_range
        )
        ask_depth = sum(
            Decimal(str(a['size'])) 
            for a in asks 
            if Decimal(str(a['price'])) <= mid_price + depth_range
        )
        order_depth = bid_depth + ask_depth
        
        # Get historical data from Gamma API price history
        price_history = stats.get('price_history', [])
        volatility = self._calculate_volatility(price_history)
        avg_price_24h = self._calculate_avg_price(price_history)
        price_range_24h = self._calculate_price_range(price_history)
        
        # Get liquidity from market data
        liquidity = Decimal(str(market.get('liquidity', 0)))
        
        # Volume from stats
        volume_24h = Decimal(str(stats.get('volume_24h', 0)))
        
        # Time to resolution
        end_date = market.get('end_date_iso')
        days_to_resolution = None
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                days_to_resolution = (end_dt - datetime.utcnow()).days
            except:
                pass
        
        return {
            'mid_price': mid_price,
            'spread': spread,
            'liquidity': liquidity,
            'volume_24h': volume_24h,
            'volatility': volatility,
            'order_depth': order_depth,
            'avg_price_24h': avg_price_24h,
            'price_range_24h': price_range_24h,
            'days_to_resolution': days_to_resolution,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth
        }
    
    def _default_metrics(self) -> Dict:
        """Return default metrics for invalid markets"""
        return {
            'mid_price': Decimal('0.5'),
            'spread': Decimal('0'),
            'liquidity': Decimal('0'),
            'volume_24h': Decimal('0'),
            'volatility': Decimal('1'),
            'order_depth': Decimal('0'),
            'avg_price_24h': Decimal('0.5'),
            'price_range_24h': Decimal('0'),
            'days_to_resolution': None,
            'bid_depth': Decimal('0'),
            'ask_depth': Decimal('0')
        }
    
    def _calculate_volatility(self, price_history: List[Dict]) -> Decimal:
        """Calculate price volatility from Gamma API historical data"""
        if len(price_history) < 2:
            return Decimal('0')
        
        # Gamma API returns price data with 'price' or 'p' field
        prices = []
        for p in price_history[-24:]:  # Last 24 data points
            price_val = p.get('price') or p.get('p')
            if price_val:
                prices.append(Decimal(str(price_val)))
        
        if len(prices) < 2:
            return Decimal('0')
        
        mean_price = sum(prices) / len(prices)
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        
        return variance.sqrt() if variance > 0 else Decimal('0')
    
    def _calculate_avg_price(self, price_history: List[Dict]) -> Decimal:
        """Calculate average price over last 24h from Gamma API data"""
        if not price_history:
            return Decimal('0.5')
        
        prices = []
        for p in price_history[-24:]:
            price_val = p.get('price') or p.get('p')
            if price_val:
                prices.append(Decimal(str(price_val)))
        
        return sum(prices) / len(prices) if prices else Decimal('0.5')
    
    def _calculate_price_range(self, price_history: List[Dict]) -> Decimal:
        """Calculate price range (max - min) over last 24h from Gamma API data"""
        if not price_history:
            return Decimal('0')
        
        prices = []
        for p in price_history[-24:]:
            price_val = p.get('price') or p.get('p')
            if price_val:
                prices.append(Decimal(str(price_val)))
        
        return max(prices) - min(prices) if prices else Decimal('0')
    
    def _detect_price_spike(self, market_id: str, current_price: Decimal) -> bool:
        """Detect if price has spiked abnormally"""
        history = self.price_history.get(market_id, [])
        
        if len(history) < 5:
            # Store current price
            self.price_history[market_id].append((datetime.utcnow(), current_price))
            return False
        
        # Get recent average (last 5 data points)
        recent_prices = [p for _, p in history[-5:]]
        avg_price = sum(recent_prices) / len(recent_prices)
        
        # Check if current price deviates significantly
        deviation = abs(current_price - avg_price) / avg_price if avg_price > 0 else Decimal('0')
        
        # Update history
        self.price_history[market_id].append((datetime.utcnow(), current_price))
        
        # Keep only last 50 data points
        if len(self.price_history[market_id]) > 50:
            self.price_history[market_id] = self.price_history[market_id][-50:]
        
        # Spike detected if deviation > 5%
        return deviation > Decimal('0.05')
    
    def _score_high_spread(self, metrics: Dict) -> float:
        """Score high spread opportunity (0-1)"""
        score = 0.0
        
        # Spread contribution (0-0.4)
        spread_pct = float(metrics['spread']) * 100
        score += min(0.4, spread_pct / 10)  # Max at 10% spread
        
        # Liquidity contribution (0-0.3)
        liquidity = float(metrics['liquidity'])
        score += min(0.3, liquidity / 1000000)  # Max at $1M liquidity
        
        # Volume contribution (0-0.2)
        volume = float(metrics['volume_24h'])
        score += min(0.2, volume / 500000)  # Max at $500k volume
        
        # Low volatility bonus (0-0.1)
        vol = float(metrics['volatility'])
        score += max(0, 0.1 - vol * 2)
        
        return score
    
    def _score_stable_liquid(self, metrics: Dict) -> float:
        """Score stable, liquid market opportunity (0-1)"""
        score = 0.0
        
        # Liquidity contribution (0-0.4)
        liquidity = float(metrics['liquidity'])
        score += min(0.4, liquidity / 1000000)
        
        # Low volatility contribution (0-0.3)
        vol = float(metrics['volatility'])
        score += max(0, 0.3 - vol * 6)
        
        # Volume contribution (0-0.2)
        volume = float(metrics['volume_24h'])
        score += min(0.2, volume / 300000)
        
        # Order depth contribution (0-0.1)
        depth = float(metrics['order_depth'])
        score += min(0.1, depth / 100000)
        
        return score
    
    def _score_mean_reversion(self, metrics: Dict) -> float:
        """Score mean reversion opportunity (0-1)"""
        score = 0.0
        
        current = float(metrics['mid_price'])
        avg = float(metrics['avg_price_24h'])
        
        if avg == 0:
            return 0.0
        
        # Deviation from mean (0-0.5)
        deviation = abs(current - avg) / avg
        score += min(0.5, deviation * 5)
        
        # Liquidity contribution (0-0.3)
        liquidity = float(metrics['liquidity'])
        score += min(0.3, liquidity / 500000)
        
        # Volume contribution (0-0.2)
        volume = float(metrics['volume_24h'])
        score += min(0.2, volume / 200000)
        
        return score
    
    def _score_low_volatility(self, metrics: Dict) -> float:
        """Score low volatility opportunity (0-1)"""
        score = 0.0
        
        # Low volatility contribution (0-0.5)
        vol = float(metrics['volatility'])
        score += max(0, 0.5 - vol * 25)
        
        # Liquidity contribution (0-0.3)
        liquidity = float(metrics['liquidity'])
        score += min(0.3, liquidity / 500000)
        
        # Volume contribution (0-0.2)
        volume = float(metrics['volume_24h'])
        score += min(0.2, volume / 200000)
        
        return score
    
    async def get_top_opportunities(self, 
                                   limit: int = 10,
                                   opportunity_types: Optional[List[OpportunityType]] = None) -> List[MarketOpportunity]:
        """Get top opportunities, optionally filtered by type"""
        
        # Refresh if stale (older than 5 minutes)
        if not self.last_scan or (datetime.utcnow() - self.last_scan).seconds > 300:
            await self.scan_all_markets()
        
        opportunities = self.opportunity_cache
        
        # Filter by type if specified
        if opportunity_types:
            opportunities = [
                opp for opp in opportunities 
                if opp.opportunity_type in opportunity_types
            ]
        
        return opportunities[:limit]
    
    def get_opportunity_summary(self) -> Dict:
        """Get summary statistics of current opportunities"""
        if not self.opportunity_cache:
            return {}
        
        summary = {
            'total_opportunities': len(self.opportunity_cache),
            'by_type': defaultdict(int),
            'avg_score': 0.0,
            'top_score': 0.0,
            'total_liquidity': Decimal('0'),
            'avg_spread': Decimal('0')
        }
        
        for opp in self.opportunity_cache:
            summary['by_type'][opp.opportunity_type.value] += 1
            summary['total_liquidity'] += opp.liquidity
            summary['avg_spread'] += opp.current_spread
        
        summary['avg_score'] = sum(o.score for o in self.opportunity_cache) / len(self.opportunity_cache)
        summary['top_score'] = self.opportunity_cache[0].score if self.opportunity_cache else 0.0
        summary['avg_spread'] = summary['avg_spread'] / len(self.opportunity_cache)
        
        return summary



    """Interface with Polymarket API"""
    
    BASE_URL = "https://clob.polymarket.com"
    
    def __init__(self, api_key: str, private_key: str):
        self.api_key = '019bdf57-30f0-763f-8e4b-f35d3e7778e7'
        self.private_key = 'RpY6Gx5uuLA18mA9WKUq6gauULTRzbxVzInqxWW5hKM='
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_markets(self, min_liquidity: Decimal = Decimal('100000')) -> List[Dict]:
        """Fetch eligible markets based on criteria"""
        # Simulated response - replace with actual API call
        async with self.session.get(f"{self.BASE_URL}/markets") as resp:
            markets = await resp.json()
            
        # Filter markets based on criteria
        eligible = []
        for market in markets:
            if (Decimal(market.get('liquidity', 0)) >= min_liquidity and
                market.get('closed', False) is False and
                market.get('outcome_count', 0) == 2):  # Binary markets only
                eligible.append(market)
                
        return eligible
    
    async def get_orderbook(self, token_id: str) -> Dict:
        """Get current orderbook for a token from CLOB API
        
        Endpoint: GET /book
        Params:
            - token_id: the token ID to get orderbook for
        """
        try:
            params = {'token_id': token_id}
            url = f"{self.CLOB_URL}/book"
            
            async with self.session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Error fetching orderbook: {resp.status}")
                    return {'bids': [], 'asks': []}
                
                data = await resp.json()
                
                # CLOB returns {bids: [...], asks: [...]}
                # Each order: {price: "0.52", size: "100.5"}
                return data
                
        except Exception as e:
            logger.error(f"Error fetching orderbook for {token_id}: {e}")
            return {'bids': [], 'asks': []}
    
    async def get_market_orderbook(self, market: Dict) -> Dict:
        """Get combined orderbook for a binary market
        
        For binary markets, we need to get the orderbook for the YES token
        Returns normalized orderbook with best bid/ask
        """
        try:
            tokens = market.get('tokens', [])
            if len(tokens) < 1:
                return {'bids': [], 'asks': []}
            
            # Get the first token (YES token typically)
            yes_token = tokens[0]
            token_id = yes_token.get('token_id')
            
            if not token_id:
                return {'bids': [], 'asks': []}
            
            return await self.get_orderbook(token_id)
            
        except Exception as e:
            logger.error(f"Error getting market orderbook: {e}")
            return {'bids': [], 'asks': []}
    
    async def get_tick_size(self) -> Decimal:
        """Get the minimum tick size for orders
        
        Endpoint: GET /tick-size
        """
        try:
            url = f"{self.CLOB_URL}/tick-size"
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return Decimal(str(data.get('tick_size', '0.01')))
        except:
            pass
        return Decimal('0.01')  # Default tick size
    
    async def get_neg_risk(self) -> bool:
        """Check if negative risk is enabled
        
        Endpoint: GET /neg-risk
        """
        try:
            url = f"{self.CLOB_URL}/neg-risk"
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('neg_risk', False)
        except:
            pass
        return False
    
    async def place_order(self, token_id: str, side: str, price: Decimal, 
                         size: Decimal, order_type: str = "GTC") -> Dict:
        """Place an order on the CLOB
        
        Note: This requires proper signature with private key
        For production, implement full order signing logic
        
        Endpoint: POST /order
        Body:
            - token_id: token to trade
            - price: limit price
            - size: order size
            - side: BUY or SELL
            - type: GTC (Good Till Cancelled), FOK (Fill or Kill), etc.
        """
        if not self.private_key:
            logger.warning("No private key - cannot place real orders (demo mode)")
            return {
                'order_id': f"demo_{token_id}_{datetime.utcnow().timestamp()}",
                'status': 'demo',
                'message': 'Demo order - not placed on exchange'
            }
        
        try:
            order = {
                "token_id": token_id,
                "price": str(price),
                "size": str(size),
                "side": side.upper(),
                "type": order_type,
                "timestamp": int(datetime.utcnow().timestamp() * 1000)
            }
            
            # In production: Sign order with private key here
            # signed_order = self._sign_order(order)
            
            url = f"{self.CLOB_URL}/order"
            async with self.session.post(url, json=order) as resp:
                if resp.status not in [200, 201]:
                    error_msg = await resp.text()
                    logger.error(f"Order placement failed: {error_msg}")
                    return {'error': error_msg}
                
                return await resp.json()
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {'error': str(e)}
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order
        
        Endpoint: DELETE /order/{order_id}
        """
        if not self.private_key:
            logger.info(f"Demo mode - simulated cancel of {order_id}")
            return True
        
        try:
            url = f"{self.CLOB_URL}/order/{order_id}"
            async with self.session.delete(url) as resp:
                return resp.status in [200, 204]
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of an order
        
        Endpoint: GET /order/{order_id}
        """
        try:
            url = f"{self.CLOB_URL}/order/{order_id}"
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
        except Exception as e:
            logger.error(f"Error fetching order status: {e}")
            return None
    
    async def get_user_orders(self, user_address: str) -> List[Dict]:
        """Get all orders for a user
        
        Endpoint: GET /orders
        """
        try:
            params = {'user': user_address}
            url = f"{self.CLOB_URL}/orders"
            
            async with self.session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                return await resp.json()
        except Exception as e:
            logger.error(f"Error fetching user orders: {e}")
            return []


class MarketMaker:
    """Core market making logic"""
    
    def __init__(self, api: PolymarketAPI, risk_manager: RiskManager):
        self.api = api
        self.risk_manager = risk_manager
        self.active_orders: Dict[str, List[str]] = {}  # market_id -> order_ids
        
    def calculate_quotes(self, orderbook: Dict, spread_bps: int, 
                        inventory_skew: Decimal) -> tuple[Decimal, Decimal]:
        """Calculate bid/ask prices based on market conditions"""
        # Get mid price
        best_bid = Decimal(orderbook.get('bids', [{}])[0].get('price', 0.5))
        best_ask = Decimal(orderbook.get('asks', [{}])[0].get('price', 0.5))
        mid_price = (best_bid + best_ask) / 2
        
        # Base spread
        spread = Decimal(spread_bps) / Decimal(10000)
        half_spread = spread / 2
        
        # Adjust for inventory - skew quotes to rebalance
        inventory_adjustment = inventory_skew * Decimal('0.01')  # 1% max adjustment
        
        bid_price = mid_price - half_spread + inventory_adjustment
        ask_price = mid_price + half_spread + inventory_adjustment
        
        # Clamp to valid range [0.01, 0.99]
        bid_price = max(Decimal('0.01'), min(Decimal('0.99'), bid_price))
        ask_price = max(Decimal('0.01'), min(Decimal('0.99'), ask_price))
        
        return bid_price, ask_price
    
    async def quote_market(self, market_id: str, config: MarketConfig):
        """Post two-sided quotes on a market"""
        try:
            # Get current orderbook
            orderbook = await self.api.get_orderbook(market_id)
            
            # Get position info
            position = self.risk_manager.positions.get(market_id)
            inventory_skew = position.inventory_skew if position else Decimal(0)
            
            # Calculate quote prices
            bid_price, ask_price = self.calculate_quotes(
                orderbook, 
                config.spread_bps,
                inventory_skew
            )
            
            # Calculate position sizes
            position_size = min(
                config.max_position_size,
                self.risk_manager.max_position_size
            )
            
            # Check if we can place orders
            if not self.risk_manager.can_open_position(market_id, position_size):
                logger.info(f"Risk limits prevent quoting {market_id}")
                return
            
            # Cancel existing orders
            await self.cancel_market_orders(market_id)
            
            # Place new orders
            orders = []
            
            # Buy order (NO side - profit if outcome is NO)
            buy_order = await self.api.place_order(
                market_id=market_id,
                side="NO",
                price=Decimal(1) - ask_price,  # Inverse for NO side
                size=position_size
            )
            orders.append(buy_order['order_id'])
            
            # Sell order (YES side)
            sell_order = await self.api.place_order(
                market_id=market_id,
                side="YES",
                price=bid_price,
                size=position_size
            )
            orders.append(sell_order['order_id'])
            
            self.active_orders[market_id] = orders
            logger.info(f"Quoted {market_id}: Bid={bid_price:.4f}, Ask={ask_price:.4f}")
            
        except Exception as e:
            logger.error(f"Error quoting market {market_id}: {e}")
    
    async def cancel_market_orders(self, market_id: str):
        """Cancel all orders for a market"""
        if market_id in self.active_orders:
            for order_id in self.active_orders[market_id]:
                try:
                    await self.api.cancel_order(order_id)
                except Exception as e:
                    logger.error(f"Error canceling order {order_id}: {e}")
            self.active_orders[market_id] = []
    
    async def rebalance_position(self, market_id: str):
        """Rebalance skewed inventory"""
        position = self.risk_manager.positions.get(market_id)
        if not position:
            return
            
        skew = position.inventory_skew
        if abs(skew) < Decimal('0.3'):
            return
            
        # Determine which side to reduce
        orderbook = await self.api.get_orderbook(market_id)
        
        if skew > 0:  # Too much YES, sell YES
            best_bid = Decimal(orderbook['bids'][0]['price'])
            reduce_size = position.yes_shares * Decimal('0.5')
            await self.api.place_order(
                market_id=market_id,
                side="YES",
                price=best_bid,
                size=reduce_size
            )
            logger.info(f"Rebalancing {market_id}: Selling {reduce_size} YES")
        else:  # Too much NO, sell NO
            best_ask = Decimal(orderbook['asks'][0]['price'])
            reduce_size = position.no_shares * Decimal('0.5')
            await self.api.place_order(
                market_id=market_id,
                side="NO",
                price=Decimal(1) - best_ask,
                size=reduce_size
            )
            logger.info(f"Rebalancing {market_id}: Selling {reduce_size} NO")


class TradingBot:
    """Main bot orchestrator with automatic market scanning"""
    
    def __init__(self, api_key: str, private_key: str, capital: Decimal):
        self.api_key = api_key
        self.private_key = private_key
        self.risk_manager = RiskManager(capital)
        self.market_maker: Optional[MarketMaker] = None
        self.market_scanner: Optional[MarketScanner] = None
        self.running = False
        self.active_markets: Dict[str, MarketConfig] = {}
        
    async def initialize(self):
        """Initialize bot components"""
        self.api = PolymarketAPI(self.api_key, self.private_key)
        await self.api.__aenter__()
        self.market_maker = MarketMaker(self.api, self.risk_manager)
        self.market_scanner = MarketScanner(self.api)
        logger.info("Bot initialized successfully with automatic market scanning")
    
    async def select_markets_dynamically(self, max_markets: int = 15) -> List[MarketConfig]:
        """Dynamically select best markets based on scanner results"""
        logger.info("Scanning for best trading opportunities...")
        
        # Get top opportunities across all types
        opportunities = await self.market_scanner.get_top_opportunities(
            limit=max_markets * 2  # Get more than needed for filtering
        )
        
        if not opportunities:
            logger.warning("No opportunities found, using fallback selection")
            return await self.select_markets_fallback()
        
        # Display opportunity summary
        summary = self.market_scanner.get_opportunity_summary()
        logger.info(f"Opportunity Summary: {summary}")
        
        # Convert opportunities to market configs
        configs = []
        for opp in opportunities[:max_markets]:
            # Adjust spread based on opportunity type
            spread_bps = self._calculate_optimal_spread(opp)
            
            config = MarketConfig(
                condition_id=opp.market_id,
                min_liquidity=max(Decimal('50000'), opp.liquidity * Decimal('0.5')),
                max_position_size=self._calculate_position_size(opp),
                spread_bps=spread_bps,
                rebalance_threshold=Decimal('0.3')
            )
            configs.append(config)
            
            logger.info(
                f"Selected: {opp.market_name[:50]}... "
                f"Type: {opp.opportunity_type.value}, "
                f"Score: {opp.score:.2f}, "
                f"Spread: {spread_bps}bps"
            )
        
        # Update active markets tracking
        self.active_markets = {c.condition_id: c for c in configs}
        
        return configs
    
    def _calculate_optimal_spread(self, opp: MarketOpportunity) -> int:
        """Calculate optimal spread based on opportunity characteristics"""
        base_spread = 150  # 1.5% base
        
        if opp.opportunity_type == OpportunityType.HIGH_SPREAD:
            # Can use wider spreads when market already has wide spreads
            return min(300, int(float(opp.current_spread) * 10000 * 0.6))
        
        elif opp.opportunity_type == OpportunityType.HIGH_LIQUIDITY:
            # Tighter spreads in liquid markets for faster fills
            return 100  # 1%
        
        elif opp.opportunity_type == OpportunityType.LOW_VOLATILITY:
            # Very tight spreads in stable markets
            return 80  # 0.8%
        
        elif opp.opportunity_type == OpportunityType.MEAN_REVERSION:
            # Medium spreads for mean reversion
            return 120  # 1.2%
        
        return base_spread
    
    def _calculate_position_size(self, opp: MarketOpportunity) -> Decimal:
        """Calculate position size based on opportunity quality"""
        base_size = self.risk_manager.max_position_size
        
        # Adjust based on score (higher score = larger position)
        score_multiplier = Decimal(str(opp.score))
        
        # Adjust based on liquidity
        liquidity_factor = min(Decimal('1.5'), opp.liquidity / Decimal('500000'))
        
        # Adjust based on volatility (lower vol = larger position)
        volatility_factor = max(Decimal('0.5'), Decimal('1') - opp.volatility * 10)
        
        adjusted_size = base_size * score_multiplier * liquidity_factor * volatility_factor
        
        # Cap at 5% of capital
        max_size = self.risk_manager.total_capital * Decimal('0.05')
        
        return min(adjusted_size, max_size)
    
    async def select_markets_fallback(self) -> List[MarketConfig]:
        """Fallback market selection if scanner fails"""
        markets = await self.api.get_markets(min_liquidity=Decimal('100000'))
        
        configs = []
        for market in markets[:10]:
            config = MarketConfig(
                condition_id=market['condition_id'],
                min_liquidity=Decimal('100000'),
                max_position_size=self.risk_manager.max_position_size,
                spread_bps=150,
                rebalance_threshold=Decimal('0.3')
            )
            configs.append(config)
            
        return configs
    
    async def monitor_opportunities(self):
        """Continuously monitor for new opportunities"""
        while self.running:
            try:
                # Rescan markets every 5 minutes
                await asyncio.sleep(300)
                
                logger.info("Refreshing market opportunities...")
                await self.market_scanner.scan_all_markets()
                
                # Get current best opportunities
                new_opportunities = await self.market_scanner.get_top_opportunities(limit=20)
                
                # Check if we should add new markets
                await self._evaluate_market_rotation(new_opportunities)
                
            except Exception as e:
                logger.error(f"Error in opportunity monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_market_rotation(self, new_opportunities: List[MarketOpportunity]):
        """Evaluate if we should rotate to better markets"""
        current_market_ids = set(self.active_markets.keys())
        
        # Find new high-scoring opportunities not currently traded
        new_high_score = [
            opp for opp in new_opportunities 
            if opp.market_id not in current_market_ids and opp.score > 0.75
        ]
        
        if not new_high_score:
            return
        
        # Find lowest-performing current markets
        current_opportunities = [
            opp for opp in self.market_scanner.opportunity_cache
            if opp.market_id in current_market_ids
        ]
        
        if not current_opportunities:
            return
        
        # Sort current by score
        current_opportunities.sort()
        
        # If new opportunity significantly better than worst current market
        worst_current = current_opportunities[0]
        best_new = new_high_score[0]
        
        if best_new.score > worst_current.score + 0.15:  # 15% better
            logger.info(
                f"Market rotation: Replacing {worst_current.market_name[:30]} "
                f"(score: {worst_current.score:.2f}) with {best_new.market_name[:30]} "
                f"(score: {best_new.score:.2f})"
            )
            
            # Cancel orders on old market
            await self.market_maker.cancel_market_orders(worst_current.market_id)
            
            # Remove from active markets
            del self.active_markets[worst_current.market_id]
            
            # Add new market config
            spread_bps = self._calculate_optimal_spread(best_new)
            new_config = MarketConfig(
                condition_id=best_new.market_id,
                min_liquidity=best_new.liquidity * Decimal('0.5'),
                max_position_size=self._calculate_position_size(best_new),
                spread_bps=spread_bps,
                rebalance_threshold=Decimal('0.3')
            )
            
            self.active_markets[best_new.market_id] = new_config
    
    async def run_cycle(self, market_configs: List[MarketConfig]):
        """Execute one trading cycle"""
        for config in market_configs:
            try:
                # Quote the market
                await self.market_maker.quote_market(
                    config.condition_id,
                    config
                )
                
                # Check for rebalancing needs
                if self.risk_manager.needs_rebalancing(
                    config.condition_id, 
                    config.rebalance_threshold
                ):
                    await self.market_maker.rebalance_position(config.condition_id)
                    
            except Exception as e:
                logger.error(f"Error in cycle for {config.condition_id}: {e}")
            
            await asyncio.sleep(1)  # Rate limiting
    
    async def run(self):
        """Main bot loop with automatic market scanning"""
        self.running = True
        await self.initialize()
        
        try:
            # Initial market selection using scanner
            market_configs = await self.select_markets_dynamically(max_markets=15)
            
            # Start opportunity monitoring in background
            monitor_task = asyncio.create_task(self.monitor_opportunities())
            
            cycle_count = 0
            while self.running:
                cycle_start = datetime.utcnow()
                
                # Run trading cycle
                await self.run_cycle(list(self.active_markets.values()))
                
                # Log performance
                total_pnl = sum(p.realized_pnl for p in self.risk_manager.positions.values())
                active_positions = len(self.risk_manager.positions)
                
                logger.info(
                    f"Cycle {cycle_count} complete. "
                    f"Active Markets: {len(self.active_markets)}, "
                    f"Positions: {active_positions}, "
                    f"Total PnL: ${total_pnl:.2f}"
                )
                
                cycle_count += 1
                
                # Full rescan every 30 minutes
                if cycle_count % 60 == 0:
                    logger.info("Performing full market rescan...")
                    market_configs = await self.select_markets_dynamically(max_markets=15)
                
                await asyncio.sleep(30)  # 30 second cycle time
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown bot"""
        logger.info("Shutting down bot...")
        self.running = False
        
        # Cancel all active orders
        if self.market_maker:
            for market_id in list(self.market_maker.active_orders.keys()):
                await self.market_maker.cancel_market_orders(market_id)
        
        if self.api:
            await self.api.__aexit__(None, None, None)
        
        logger.info("Bot shutdown complete")


# Example usage
async def main():
    # Configuration
    API_KEY = "your_api_key_here"
    PRIVATE_KEY = "your_private_key_here"
    INITIAL_CAPITAL = Decimal('10000')  # $10,000
    
    # Create and run bot
    bot = TradingBot(API_KEY, PRIVATE_KEY, INITIAL_CAPITAL)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
